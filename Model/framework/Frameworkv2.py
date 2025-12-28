from graphviz import Digraph
import os

def create_final_architecture_diagram(filename='final_gw_optical_dual_stream_arch'):
    # 初始化有向图
    dot = Digraph(comment='Final GW-Optical Dual-Stream Architecture', format='png')

    # --- 全局图形属性 ---
    dot.attr(rankdir='TB') # 从上到下布局
    dot.attr(splines='ortho') # 使用正交线 (直角折线)，使图表更整洁
    dot.attr(compound='true') # 允许在子图之间连线
    dot.attr('node', fontname='Arial', fontsize='12')
    dot.attr('edge', fontname='Arial', fontsize='10')

    # --- 节点样式定义 ---
    # 颜色定义
    c_input_data = '#E0E0E0' # 普通输入数据 (浅灰)
    c_param = '#D1C4E9'      # 特殊参数/参考输入 (浅紫)
    c_module = '#BBDEFB'     # 处理模块 (浅蓝)
    c_tensor = '#FFE0B2'     # 中间张量/向量 (浅橙)
    c_op = '#FFECB3'         # 操作节点 (拼接等，浅黄)
    c_loss = '#FFCDD2'       # 损失函数 (浅红)

    # 样式字典
    node_style_input_data = {'shape': 'box', 'style': 'filled', 'fillcolor': c_input_data}
    node_style_param = {'shape': 'box', 'style': 'filled', 'fillcolor': c_param}
    node_style_module = {'shape': 'component', 'style': 'filled', 'fillcolor': c_module}
    node_style_tensor = {'shape': 'ellipse', 'style': 'filled', 'fillcolor': c_tensor}
    # 新增操作节点样式 (如拼接)
    node_style_op = {'shape': 'circle', 'style': 'filled', 'fillcolor': c_op, 'width': '0.8', 'fixedsize': 'true'}
    node_style_loss = {'shape': 'hexagon', 'style': 'filled', 'fillcolor': c_loss}

    # =============== 1. 输入层 (Inputs) ===============
    with dot.subgraph(name='cluster_inputs') as c:
        c.attr(style='invis')
        # 引力波输入
        c.node('GW_Input', 'GW Parameters\n(Scalars: mass, spin...)', **node_style_input_data)
        # [新增] Skymap 输入
        c.node('Skymap_Input', 'Skymap Sequence\n(1D MOC/HEALPix Data)', **node_style_input_data)
        
        # 光学输入
    
        c.node('Ref_Time', 'Reference Time Points\n(Parameter)', **node_style_param)
        c.node('Opt_Input', 'Optical Data\n(Light Curves)', **node_style_input_data)
        c.node('CLS_Token', 'Learnable CLS Token\n(Parameter)', **node_style_param)
        

    # =============== 2. 编码器层 (Encoders) ===============
    # 使用一个不可见的 cluster 来组合两个大的编码器模块，确保它们并排显示
    with dot.subgraph(name='cluster_encoders_container') as c:
        c.attr(style='invis')

        # --- [核心修改] 双流引力波编码器子图 ---
        with c.subgraph(name='cluster_gw_dual_encoder') as gw:
            gw.attr(label='Dual-Stream GW Encoder', style='dashed', color='gray', bgcolor='white')
            
            # 2.1 标量分支
            gw.node('GW_Scalar_MLP', 'Scalar Encoder\n(MLP)', **node_style_module)
            
            # 2.2 Skymap 分支
            gw.node('GW_Skymap_ResNet', 'Skymap Encoder\n(1D ResNet)', **node_style_module)
            
            # 2.3 融合部分
            # 使用较小的圆形节点表示拼接操作
            gw.node('GW_Concat', 'Concat', **node_style_op)
            gw.node('GW_Fusion_MLP', 'Fusion MLP', **node_style_module)

            # 内部连接
            gw.edge('GW_Scalar_MLP', 'GW_Concat')
            gw.edge('GW_Skymap_ResNet', 'GW_Concat')
            gw.edge('GW_Concat', 'GW_Fusion_MLP')

        # --- 光学编码器 (保持不变) ---
        c.node('Opt_Enc', 'Optical Encoder\n(mTAN + CLS Injection)', **node_style_module)

    # =============== 3. 中间张量 (Intermediate Tensors) ===============
    with dot.subgraph(name='cluster_tensors') as c:
        c.attr(style='invis')
        # 引力波向量现在来自 Fusion MLP 的输出
        c.node('Vec_g', 'Final GW Vector\n[ g ]', **node_style_tensor)
        c.node('Mat_Hl', 'Optical Matrix\n[ H_l ]\n((N+1) x J)', **node_style_tensor)

    # =============== 4. 对齐分支 (Alignment Branch) ===============
    with dot.subgraph(name='cluster_alignment') as c:
        c.attr(label='Alignment Branch (Contrastive Learning)', style='dashed', color='blue')
        c.node('Proj_GW', 'Projection Head\n(GW)', **node_style_module)
        c.node('Proj_Opt', 'Projection Head\n(Optical)', **node_style_module)
        c.node('z_g', 'z_g\n(Normalized)', **node_style_tensor)
        c.node('z_l', 'z_l\n(Normalized)', **node_style_tensor)
        c.node('Loss_ITC', 'Alignment Loss\n(L_itc)', **node_style_loss)

    # =============== 5. 融合分支 (Fusion Branch) ===============
    with dot.subgraph(name='cluster_fusion') as c:
        c.attr(label='Fusion Branch (Classification)', style='dashed', color='red')
        c.node('Cross_Attn', 'Cross Attention\n(Fusion Block)', **node_style_module)
        c.node('Fused_Vec', 'Fused Feature', **node_style_tensor)
        c.node('Classifier', 'Classifier MLP', **node_style_module)
        c.node('Loss_CLS', 'Classification Loss\n(L_cls)', **node_style_loss)

    # =============== 6. 定义外部连接 (Edges) ===============
    
    # --- 输入 -> 编码器 ---
    # [修改] 引力波标量 -> 标量编码器
    dot.edge('GW_Input', 'GW_Scalar_MLP')
    # [新增] Skymap 序列 -> ResNet 编码器
    dot.edge('Skymap_Input', 'GW_Skymap_ResNet')
    
    # 光学输入 -> 光学编码器
    dot.edge('Opt_Input', 'Opt_Enc')
    dot.edge('Ref_Time', 'Opt_Enc')
    dot.edge('CLS_Token', 'Opt_Enc')
    
    # --- 编码器 -> 中间张量 ---
    # [修改] 双流编码器的最终输出 (Fusion MLP) -> Vec_g
    dot.edge('GW_Fusion_MLP', 'Vec_g')
    dot.edge('Opt_Enc', 'Mat_Hl')

    # --- 张量 -> 对齐分支 ---
    dot.edge('Vec_g', 'Proj_GW')
    dot.edge('Proj_GW', 'z_g')
    dot.edge('z_g', 'Loss_ITC')
    
    # 提取 CLS 向量 (Index 0)
    dot.edge('Mat_Hl', 'Proj_Opt', label='Index 0\n(CLS Vector)', fontcolor='blue')
    dot.edge('Proj_Opt', 'z_l')
    dot.edge('z_l', 'Loss_ITC')

    # --- 张量 -> 融合分支 ---
    # GW 向量作为 Query
    dot.edge('Vec_g', 'Cross_Attn', label='Query (g)', fontcolor='red')
    # 提取时序特征 (Index 1:N) 作为 Key/Value
    dot.edge('Mat_Hl', 'Cross_Attn', label='Key/Value\n(Time Series)', fontcolor='red')
    
    dot.edge('Cross_Attn', 'Fused_Vec')
    dot.edge('Fused_Vec', 'Classifier')
    dot.edge('Classifier', 'Loss_CLS')

    # 生成并保存图像
    try:
        # format='png' 或 'pdf', 'svg' 等
        output_path = dot.render(filename, view=True, cleanup=True)
        print(f"流程图已成功生成: {output_path}")
    except Exception as e:
        print(f"生成流程图时出错: {e}")
        print("请确保已安装 Graphviz 软件并将其添加到系统路径中 (不仅是 pip install graphviz)。")

if __name__ == '__main__':
    create_final_architecture_diagram(filename='/fred/oz016/bgao_kn/ML+GW+KN/Model/framework/gw_optical_architecture_v2.png')