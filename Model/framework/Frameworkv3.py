from graphviz import Digraph


def create_framework_v3_diagram(filename='gw_optical_architecture_v3'):
    """
    Generate framework_v3 diagram based on the current GW-Optical ALBEF model:
    - Dual-stream GW encoder (scalar + skymap)
    - Optical encoder with spatial embedding + CLS injection
    - Dual cross-attention fusion with credible level
    """
    dot = Digraph(comment='GW-Optical ALBEF Framework v3', format='png')

    # Global graph attributes
    dot.attr(rankdir='TB')
    dot.attr(splines='ortho')
    dot.attr(compound='true')
    dot.attr('node', fontname='Arial', fontsize='12')
    dot.attr('edge', fontname='Arial', fontsize='10')

    # Colors
    c_input_data = '#E0E0E0'
    c_param = '#D1C4E9'
    c_module = '#BBDEFB'
    c_tensor = '#FFE0B2'
    c_op = '#FFECB3'
    c_loss = '#FFCDD2'
    c_embed = '#C8E6C9'

    # Styles
    node_style_input_data = {'shape': 'box', 'style': 'filled', 'fillcolor': c_input_data}
    node_style_param = {'shape': 'box', 'style': 'filled', 'fillcolor': c_param}
    node_style_module = {'shape': 'component', 'style': 'filled', 'fillcolor': c_module}
    node_style_tensor = {'shape': 'ellipse', 'style': 'filled', 'fillcolor': c_tensor}
    node_style_op = {'shape': 'circle', 'style': 'filled', 'fillcolor': c_op, 'width': '0.8', 'fixedsize': 'true'}
    node_style_loss = {'shape': 'hexagon', 'style': 'filled', 'fillcolor': c_loss}
    node_style_embed = {'shape': 'component', 'style': 'filled', 'fillcolor': c_embed}

    # =============== 1. Inputs ===============
    with dot.subgraph(name='cluster_inputs') as c:
        c.attr(style='invis')
        c.node('GW_Input', 'GW Scalars\n(mass, spin, ...)', **node_style_input_data)
        c.node('Skymap_Input', 'Skymap Sequence\n(MOC/HEALPix)', **node_style_input_data)
        c.node('Opt_Input', 'Optical Light Curves', **node_style_input_data)
        c.node('Opt_Coord', 'Optical Coordinates\n(RA, Dec)', **node_style_input_data)
        c.node('Ref_Time', 'Reference Time Points\n(Parameter)', **node_style_param)
        c.node('CLS_Token', 'Learnable CLS Token\n(Parameter)', **node_style_param)

    # =============== 2. Encoders ===============
    with dot.subgraph(name='cluster_encoders_container') as c:
        c.attr(style='invis')

        # --- GW dual-stream encoder ---
        with c.subgraph(name='cluster_gw_dual_encoder') as gw:
            gw.attr(label='Dual-Stream GW Encoder', style='dashed', color='gray', bgcolor='white')
            gw.node('GW_Scalar_MLP', 'Scalar Encoder\n(MLP)', **node_style_module)
            gw.node('GW_Skymap_Enc', 'Skymap Encoder\n(1D ResNet / Lightweight)', **node_style_module)
            gw.node('GW_Concat', 'Concat', **node_style_op)
            gw.node('GW_Fusion_MLP', 'Fusion MLP', **node_style_module)
            gw.node('GW_Skymap_Feat', 'Skymap Feature Map', **node_style_tensor)
            gw.node('GW_Seq_Proj', 'Seq Projection', **node_style_module)

            gw.edge('GW_Scalar_MLP', 'GW_Concat')
            gw.edge('GW_Skymap_Enc', 'GW_Concat')
            gw.edge('GW_Concat', 'GW_Fusion_MLP')
            gw.edge('GW_Skymap_Enc', 'GW_Skymap_Feat')
            gw.edge('GW_Skymap_Feat', 'GW_Seq_Proj')

        # --- Optical encoder ---
        with c.subgraph(name='cluster_opt_encoder') as opt:
            opt.attr(label='Optical Encoder', style='dashed', color='gray', bgcolor='white')
            opt.node('Opt_Spatial_Embed', 'Spatial Embedding\n(RA/Dec MLP)', **node_style_embed)
            opt.node('Opt_CLS_Add', 'Add', **node_style_op)
            opt.node('Opt_Enc', 'mTAN Encoder\n+ CLS Injection', **node_style_module)

            opt.edge('Opt_Spatial_Embed', 'Opt_CLS_Add')
            opt.edge('Opt_CLS_Add', 'Opt_Enc')

    # =============== 3. Intermediate Tensors ===============
    with dot.subgraph(name='cluster_tensors') as c:
        c.attr(style='invis')
        c.node('Vec_g', 'GW Global Vector\n[ g ]', **node_style_tensor)
        c.node('H_gw', 'GW Spatial Map\n[ H_gw ]', **node_style_tensor)
        c.node('z_l', 'Optical CLS Vector\n[ z_l ]', **node_style_tensor)
        c.node('H_l', 'Optical Matrix\n[ H_l ]\n(N x J)', **node_style_tensor)

    # =============== 4. Alignment Branch ===============
    with dot.subgraph(name='cluster_alignment') as c:
        c.attr(label='Alignment Branch (Contrastive)', style='dashed', color='blue')
        c.node('Proj_GW', 'Projection Head\n(GW)', **node_style_module)
        c.node('Proj_Opt', 'Projection Head\n(Optical)', **node_style_module)
        c.node('z_g', 'z_g\n(Normalized)', **node_style_tensor)
        c.node('z_l_norm', 'z_l\n(Normalized)', **node_style_tensor)
        c.node('Loss_ITC', 'Alignment Loss\n(L_itc)', **node_style_loss)

    # =============== 5. Fusion Branch ===============
    with dot.subgraph(name='cluster_fusion') as c:
        c.attr(label='Fusion Branch (Dual Cross-Attention)', style='dashed', color='red')
        c.node('Cross_G2O', 'Cross Attention\n(GW -> Optical)', **node_style_module)
        c.node('Cross_O2G', 'Cross Attention\n(Optical -> GW)', **node_style_module)
        c.node('Fused_Opt', 'Fused Optical Feature', **node_style_tensor)
        c.node('Fused_GW', 'Fused GW Feature', **node_style_tensor)
        c.node('Cred_Level', 'Credible Level\n(Skymap + RA/Dec)', **node_style_tensor)
        c.node('Fuse_Concat', 'Concat', **node_style_op)
        c.node('Fused_Vec', 'Fused Feature', **node_style_tensor)
        c.node('Classifier', 'Classifier MLP', **node_style_module)
        c.node('Loss_CLS', 'Classification Loss\n(L_cls)', **node_style_loss)

    # =============== 6. External Connections ===============
    # Inputs -> Encoders
    dot.edge('GW_Input', 'GW_Scalar_MLP')
    dot.edge('Skymap_Input', 'GW_Skymap_Enc')

    dot.edge('Opt_Input', 'Opt_Enc')
    dot.edge('Ref_Time', 'Opt_Enc')
    dot.edge('Opt_Coord', 'Opt_Spatial_Embed')
    dot.edge('CLS_Token', 'Opt_CLS_Add')

    # Encoders -> Tensors
    dot.edge('GW_Fusion_MLP', 'Vec_g')
    dot.edge('GW_Seq_Proj', 'H_gw')
    dot.edge('Opt_Enc', 'z_l')
    dot.edge('Opt_Enc', 'H_l')

    # Alignment Branch
    dot.edge('Vec_g', 'Proj_GW')
    dot.edge('Proj_GW', 'z_g')
    dot.edge('z_g', 'Loss_ITC')

    dot.edge('z_l', 'Proj_Opt')
    dot.edge('Proj_Opt', 'z_l_norm')
    dot.edge('z_l_norm', 'Loss_ITC')

    # Fusion Branch: dual cross-attention
    dot.edge('Vec_g', 'Cross_G2O', label='Query (g)', fontcolor='red')
    dot.edge('H_l', 'Cross_G2O', label='Key/Value\n(Time Series)', fontcolor='red')

    dot.edge('z_l', 'Cross_O2G', label='Query (CLS)', fontcolor='red')
    dot.edge('H_gw', 'Cross_O2G', label='Key/Value\n(Skymap Seq)', fontcolor='red')

    dot.edge('Cross_G2O', 'Fused_Opt')
    dot.edge('Cross_O2G', 'Fused_GW')

    # Credible level from Skymap + Optical coords
    dot.edge('Skymap_Input', 'Cred_Level')
    dot.edge('Opt_Coord', 'Cred_Level')

    # Concat -> Classifier
    dot.edge('Fused_Opt', 'Fuse_Concat')
    dot.edge('Fused_GW', 'Fuse_Concat')
    dot.edge('Cred_Level', 'Fuse_Concat')
    dot.edge('Fuse_Concat', 'Fused_Vec')
    dot.edge('Fused_Vec', 'Classifier')
    dot.edge('Classifier', 'Loss_CLS')

    # Render
    try:
        output_path = dot.render(filename, view=True, cleanup=True)
        print(f"Framework v3 diagram generated: {output_path}")
    except Exception as e:
        print(f"Failed to generate diagram: {e}")
        print("Ensure Graphviz is installed and on your PATH (not just pip install graphviz).")


if __name__ == '__main__':
    create_framework_v3_diagram(
        filename='/fred/oz016/bgao_kn/ML+GW+KN/Model/framework/gw_optical_architecture_v3'
    )
