from graphviz import Digraph
import os

def create_updated_architecture_diagram(filename='updated_gw_optical_architecture'):
    # 初始化有向图
    dot = Digraph(comment='Updated GW-Optical ALBEF Architecture', format='png')

    # --- 全局图形属性 ---
    # TB: Top to Bottom 布局
    dot.attr(rankdir='TB')
    # ortho: 正交连线 (直角折线)
    dot.attr(splines='ortho')
    # 设置默认字体
    dot.attr('node', fontname='Arial', fontsize='12')
    dot.attr('edge', fontname='Arial', fontsize='10')

    # --- 节点样式定义 ---
    c_input = '#E0E0E0'      # 输入数据 (浅灰)
    c_module = '#BBDEFB'     # 处理模块 (浅蓝)
    c_tensor = '#FFE0B2'     # 中间张量/向量 (浅橙)
    c_loss = '#FFCDD2'       # 损失函数 (浅红)
    c_param = '#D1C4E9'      # 特殊参数/参考输入 (浅紫)

    # 定义不同类型节点的属性字典
    node_style_input = {'shape': 'box', 'style': 'filled', 'fillcolor': c_input}
    node_style_param = {'shape': 'box', 'style': 'filled', 'fillcolor': c_param}
    # 'component' 形状表示功能模块
    node_style_module = {'shape': 'component', 'style': 'filled', 'fillcolor': c_module}
    # 'ellipse' 形状表示数据张量
    node_style_tensor = {'shape': 'ellipse', 'style': 'filled', 'fillcolor': c_tensor}
    # 'hexagon' 形状表示损失函数
    node_style_loss = {'shape': 'hexagon', 'style': 'filled', 'fillcolor': c_loss}

    # === 1. 输入层 (Inputs) ===
    # 使用 subgraph 将相关节点分组，cluster_ 前缀会在图中显示一个外框
    with dot.subgraph(name='cluster_inputs') as c:
        # style='invis' 隐藏子图的边框，只用于逻辑分组
        c.attr(style='invis')
        c.node('GW_Input', 'GW Parameters\n(m, s, ...)', **node_style_input)
        c.node('Ref_Time', 'Reference Time Points\nr = [r_1, ..., r_N]', **node_style_param)
        # [新增] CLS Token 输入
        c.node('CLS_Token', 'Learnable CLS Token\n(Parameter)', **node_style_param)
        c.node('Opt_Input', 'Optical Data\n(LSST Light Curves)', **node_style_input)

    # === 2. 编码器层 (Encoders) ===
    with dot.subgraph(name='cluster_encoders') as c:
        c.attr(style='invis')
        c.node('GW_Enc', 'GW Encoder\n(MLP)', **node_style_module)
        # 更新标签以反映其处理 CLS 的能力
        c.node('Opt_Enc', 'Optical Encoder\n(Multi-Time Attention)', **node_style_module)

    # === 3. 中间张量 (Intermediate Tensors) ===
    with dot.subgraph(name='cluster_tensors') as c:
        c.attr(style='invis')
        c.node('Vec_g', 'GW Vector\n[ g ]', **node_style_tensor)
        # [更新] 维度变为 (N+1) x J
        c.node('Mat_Hl', 'Optical Matrix\n[ H_l ]\n((N+1) x J)', **node_style_tensor)

    # === 4. 对齐分支 (Alignment Branch) ===
    with dot.subgraph(name='cluster_alignment') as c:
        # 设置子图标签和样式
        c.attr(label='Alignment Branch (Contrastive)', style='dashed', color='gray')
        
        c.node('Proj_GW', 'Projection Head\n(GW)', **node_style_module)
        # [移除] Attention Pooling 模块
        # c.node('Attn_Pool', 'Attention Pooling', **node_style_module)
        c.node('Proj_Opt', 'Projection Head\n(Optical)', **node_style_module)
        
        c.node('z_g', 'z_g\n(M-dim)', **node_style_tensor)
        c.node('z_l', 'z_l\n(M-dim)', **node_style_tensor)
        
        c.node('Loss_ITC', 'Alignment Loss\n(L_itc)', **node_style_loss)

    # === 5. 融合分支 (Fusion Branch) ===
    with dot.subgraph(name='cluster_fusion') as c:
        c.attr(label='Fusion Branch (Classification)', style='dashed', color='gray')
        
        c.node('Cross_Attn', 'Cross Attention\n(Fusion)', **node_style_module)
        c.node('Fused_Vec', 'Fused Vector', **node_style_tensor)
        c.node('Classifier', 'Classifier\n(MLP)', **node_style_module)
        
        c.node('Loss_CLS', 'Classification Loss\n(L_cls)', **node_style_loss)

    # === 6. 定义连接边 (Edges) ===
    
    # -> 编码器输入
    dot.edge('GW_Input', 'GW_Enc')
    dot.edge('Opt_Input', 'Opt_Enc')
    # [更新] CLS Token 和 参考时间点共同作为查询输入到 MTAN
    dot.edge('Ref_Time', 'Opt_Enc')
    dot.edge('CLS_Token', 'Opt_Enc')
    
    # 编码器 -> 张量
    dot.edge('GW_Enc', 'Vec_g')
    dot.edge('Opt_Enc', 'Mat_Hl')

    # -> 对齐分支路径
    dot.edge('Vec_g', 'Proj_GW')
    dot.edge('Proj_GW', 'z_g')
    dot.edge('z_g', 'Loss_ITC')
    
    # [关键修改] 直接从矩阵中提取索引 0 (CLS向量) 用于对齐
    # fontcolor='blue' 用于突出显示数据流标签
    dot.edge('Mat_Hl', 'Proj_Opt', label='Index 0\n(CLS Vector)', fontcolor='blue')
    dot.edge('Proj_Opt', 'z_l')
    dot.edge('z_l', 'Loss_ITC')

    # -> 融合分支路径
    # GW 向量作为 Query
    dot.edge('Vec_g', 'Cross_Attn', label='Query (g)', fontcolor='blue')
    # [关键修改] 明确使用索引 1 到 N (时序特征) 作为 Key/Value
    dot.edge('Mat_Hl', 'Cross_Attn', label='Key/Value\n(Indices 1:N)', fontcolor='blue')
    
    dot.edge('Cross_Attn', 'Fused_Vec')
    dot.edge('Fused_Vec', 'Classifier')
    dot.edge('Classifier', 'Loss_CLS')

    # 生成并保存图像
    # view=True 会在生成后自动打开图片
    try:
        dot.render(filename, view=True)
        print(f"流程图已成功生成: {filename}.png")
    except Exception as e:
        print(f"生成流程图时出错: {e}")
        print("请确保已安装 Graphviz 软件并将其添加到系统路径中。")

if __name__ == '__main__':
    # 调用函数生成流程图
    # 您可以修改 filename 参数来指定输出文件名
    create_updated_architecture_diagram('gw_optical_architecture_v1')