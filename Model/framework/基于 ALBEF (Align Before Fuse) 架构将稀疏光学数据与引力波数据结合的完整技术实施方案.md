该方案的核心思想是：利用信息丰富的引力波（GW）作为“锚点”，先在全局物理属性上与稀疏光学数据（LC）**对齐（Align）**，再利用交叉注意力机制进行细粒度的**融合验证（Fuse）**，从而在只有2-3个光度点的情况下实现高精度的千新星分类。
![[gw_optical_architecture_v2.png]]


---
### 1. 模型架构设计 (Architecture)

模型由双塔单模态编码器和一个多模态融合编码器组成。

#### A. 单模态编码器 (Unimodal Encoders)

- **引力波编码器 ($E_{GW}$):**
    - **输入:** 1D引力波参数 + MOC skymap
    - **模型:** MLP + 1D ResNet 编码器
    - **输出:** 物理特征向量 $g \in \mathbb{R}^{d_g}$ (包含质量、距离、时间等隐含信息)。
- **光学编码器 ($E_{LC}$):**
    - **输入:** 稀疏序列 $\{(t_i, m_i, \sigma_i, b_i)\}_{i=1}^L$，其中 $L \approx 10\sim 20$, N个参考时间点$r=[r_1,r_2,...,r_N]$
    - **模型:** **mTAND** (Multi-Time Attention Networks) 
        - _关键设计:_ 需要额外输入[CLS] token，具体形式通过在参考时间点进行时间嵌入后拼接一个可学习的向量，最后取该向量的mTAN输出用于对齐
    - **输出:** 序列特征 $H_L \in \mathbb{R}^{J \times N}$。

---
#### B. 投影与对齐层 (Projection & Alignment)
通过投影，将引力波编码向量$g$投影为$z_g=\phi(g)\in R^M$；

##### 光学编码器对齐向量的获取：
将输入的参考时间序列定义改为两部分：
1. **全局部分 (CLS)**：一个不依赖于具体时间 $t$ 的可学习向量 $\mathbf{q}_{cls}$。
2. **局部部分 (Ref Times)**：依赖于输入 $r$ 的标准时间嵌入 $\phi_h(r)$。
定义混合查询矩阵 $\Psi$：
$$\Psi = [\mathbf{q}_{cls}, \phi_h(r_1), \phi_h(r_2), \dots, \phi_h(r_N)]$$
- $\mathbf{q}_{cls} \in \mathbb{R}^{d_r}$：是一个**可学习的参数（Learnable Parameter）**，初始化为随机向量。它代替了公式中的 $\phi_h(t_{cls})$。
- $\phi_h(r_i)$：是常规的周期性时间嵌入。
**代入注意力核函数 $\kappa$：**
对于第一项（CLS token），注意力权重 $\kappa_{cls}$ 计算变为：
$$\kappa_{cls}(t_{id}) = \frac{\exp \left( \mathbf{q}_{cls} \mathbf{w} \mathbf{v}^T \phi_h(t_{id})^T / \sqrt{d_k} \right)}{\sum_{i'} \exp \left( \mathbf{q}_{cls} \mathbf{w} \mathbf{v}^T \phi_h(t_{i'd})^T / \sqrt{d_k} \right)}$$
**相当于在计算参考时间之外，额外引入可学习的向量$q_{cls}\in R^{d_r}$与经过时间编码后的参考时间类似。**
最终取$mTAN(cls,\mathbf{s})\in R^{J}$经过投影后得到$z_l\in R^{M}$用于对齐。

-  通过线性层将 $u$ 和 $v$（取平均池化）映射到低维超球面上，计算 **InfoNCE Loss**。
- **作用:** 强迫模型学会“物理一致性”（例如：GW显示的近距离爆发应对应较亮的光学点）。

##### 引力波（Skymap）编码器

输入引力波MOC格式天图$g_{moc} \in R^{7\times 19200}$，其中19200代表像素数量，通道数为7，分别为$[ra,dec,pixel\_area,pixel\_prob,dist\_mean,dist\_sigma,dist\_norm]$。
*Note：部分像素的Dist mean 和 Dist Norm 可能为inf，导致计算失败，这些像素源于引力波参数估计失败，直接将距离信息置为0，0，0*

**使用ResNet 结合1D卷积，将$g_{moc}$变换为$h_{skymap}\in R^{128}$, 之后与引力波参数$g_{scalars}\in R^{7}$合并之后，通过MLP投影为 $g\in R^{128}$。**

---
#### C. 多模态融合编码器 (Multimodal Encoder)
使用$g\in R^{d_g},H_L\in R^{N\times J}$计算交叉注意力
-  **输入（ Cross-Attention**）。
    - **Query ($Q$):** 来自引力波特征 $g$（作为“提问者”）。
    - **Key/Value ($K, V$):** 来自光学序列特征 $H_L$（作为“被查询的证据”）。
**维度匹配步骤**
需要三个线性投影层：$W_q, W_k, W_v$。

**第一步：序列化 Query**
注意力机制要求 Query 具有序列维度。因为 $g$ 是一个全局向量，需将其视为**长度为 1 的序列**。
- 操作：`g.unsqueeze(1)`
- Shape 变化：$[B, d_g] \rightarrow [B, 1, d_g]$

**第二步：线性投影 (Projection)**
将 $g$ 和 $H_l$ 映射到相同的特征维度 $d_{q},d_v$。
- **Query ($Q$)**: $g \cdot W_q$
    - $W_q \in \mathbb{R}^{d_g \times d_{q}}$
    - **$Q$ Shape**: $[B, 1, d_{q}]$
- **Key ($K$)**: $H_l \cdot W_k$
    - $W_k \in \mathbb{R}^{J \times d_{q}}$
    - **$K$ Shape**: $[B, N, d_{q}]$
- **Value ($V$)**: $H_l \cdot W_v$
    - $W_v \in \mathbb{R}^{J \times d_{v}}$
    - **$V$ Shape**: $[B, N, d_{v}]$

**第三步：注意力分数计算 (Attention Score)**
计算 $Q$ 与 $K$ 的点积。这里的关键是**最后维度的匹配** ($d_{q}$)。
- 公式：$Score = \frac{Q K^T}{\sqrt{d_{q}}}$
- 运算：$[B, 1, d_q] \times [B, d_{q}, N]$ (即 $K$ 的转置)
- **Attention Map Shape**: $[B, 1, N]$
    - _物理意义_：对于该引力波事件，光变曲线的 $N$ 个时间参考点中，每个点的重要性权重。

**第四步: 加权求和 (Weighted Sum)**
利用 Attention Map 对 $V$ 进行加权。
- 公式：$Output = \text{Softmax}(Score) \cdot V$
- 运算：$[B, 1, N] \times [B, N, d_{v}]$
- **Result Shape**: $[B, 1, d_{v}]$

**第五步：压缩 (Squeeze)**
为了后续分类，通常去掉序列维度。
- 操作：`result.squeeze(1)`
- **最终融合向量 Shape**: $[B, d_{v}]$v

---

### 2. 训练设计

#### 2.1 对比学习 (Alignment Branch)

此部分旨在将同一物理事件的引力波向量与光变曲线向量映射到相近的特征空间。
- **输入状态**：    
    - $z_g \in \mathbb{R}^{B \times M}$：经过投影并 $L_2$ 归一化的 GW 向量。
    - $z_l \in \mathbb{R}^{B \times M}$：经过投影并 $L_2$ 归一化的 Optical 向量（来自 MTAN 的 Global Query/CLS）。
- **采样策略**：**Batch 内唯一采样**。即一个 Batch 内包含 $B$ 个完全不同的事件，每个事件随机选取一条光变曲线。此时，相似度矩阵的**对角线为正样本**，非对角线均为负样本。
- **温度参数**：$\tau$ (可学习)。

##### 损失函数计算公式 ($L_{glc}$)

采用对称的 InfoNCE Loss (GW-LC Contrastive Loss)。
首先计算相似度矩阵 $S \in \mathbb{R}^{B \times B}$：
$$S_{i,j} = z_{g,i} \cdot z_{l,j}^\top$$

1. GW-to-Optical Loss (行方向):
$$L_{g2l} = - \frac{1}{B} \sum_{i=1}^{B} \log \frac{\exp(S_{i,i} / \tau)}{\sum_{j=1}^{B} \exp(S_{i,j} / \tau)}$$
2. Optical-to-GW Loss (列方向):
$$L_{l2g} = - \frac{1}{B} \sum_{i=1}^{B} \log \frac{\exp(S_{i,i} / \tau)}{\sum_{j=1}^{B} \exp(S_{j,i} / \tau)}$$
总对齐损失：
$$L_{glc} = \frac{1}{2} (L_{g2l} + L_{l2g})$$

---
#### 2.2 混合/融合学习 (Fusion Branch)

此部分利用交叉注意力机制，进行精细化的二分类任务（判断 $g$ 与 $H_l$ 是否匹配）。
- **输入状态**：
    - $g_{feat} \in \mathbb{R}^{B \times d_g}$：GW Encoder 的原始输出特征（Query）。
    - $H_{l\_feat} \in \mathbb{R}^{B \times N \times J}$：MTAN 的时序输出矩阵（Key/Value）。
- **采样策略**：**困难负样本挖掘 (Hard Negative Mining)**。
    - **正样本通路**：直接使用 Batch 内原本对应的配对 $(g_i, H_{l,i})$。
    - **负样本通路**：利用对齐分支的相似度矩阵 $S$，为每个 $g_i$ 采样一个最易混淆的错配光变 $H_{l,j}$（$j \neq i$）。

##### 损失函数计算公式 ($L_{glm}$)

采用标准的交叉熵损失函数（Cross Entropy）。
设分类器输出为 $p(y|g, H_l)$，其中 $y \in \{0, 1\}$。
1. 正样本损失 (Positive Pass):
输入为匹配对，目标标签 $y=1$。
$$L_{glm}^{pos} = - \frac{1}{B} \sum_{i=1}^{B} \log p(y=1 | g_i, H_{l,i})$$
2. 负样本损失 (Negative Pass):
输入为挖掘出的困难错配对 $(g_i, H_{l, neg\_idx(i)})$，目标标签 $y=0$。
$$L_{glm}^{neg} = - \frac{1}{B} \sum_{i=1}^{B} \log p(y=0 | g_i, H_{l, neg\_idx(i)})$$
总分类损失：
$$L_{glm} = \frac{1}{2} (L_{cls}^{pos} + L_{cls}^{neg})$$

---
#### 2.3 总优化目标

整个网络端到端联合训练的总损失函数为：

$$L_{total} = L_{itc} + L_{cls}$$
(注：通常权重设为 1:1 即可，也可根据收敛情况添加权重系数 $\lambda$)

---

### 3. 最终应用：分类推断 (Inference)

在实际应用中，处理流程如下：

1. **触发:** 引力波探测器 (LIGO/Virgo) 触发一个事件，给出 GW 数据。
    
2. **观测:** 光学望远镜在接下来的几天内捕捉到 2-3 个稀疏的光度点。
    
3. **输入模型:** 将两者同时送入训练好的 ALBEF 模型。
    
4. **判决:**
    - 模型输出 GLM Score (匹配分数)。
    - **如果 Score > 阈值:** 判定为千新星 (Kilonova)，且这几个光点确实是该 GW 事件的对应体。
    - **如果 Score < 阈值:** 判定为背景噪声或无关瞬变源