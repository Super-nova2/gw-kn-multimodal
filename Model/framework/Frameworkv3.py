from graphviz import Digraph


def create_framework_v5_diagram(
    filename="/fred/oz016/bgao_kn/ML+GW+KN/Model/framework/gw_optical_architecture_v5_clean"
):
    """
    Draw the current final GW-Optical ALBEF architecture.

    Current final physical_dual_hgw design:
    - Optical encoder is decoupled into curve encoder and coordinate encoder
    - GW encoder is decoupled into scalar encoder and skymap encoder
    - Contrastive branch performs fusion + projection
    - Classification branch keeps dual cross-attention
    - Pair similarity is an optional classifier input
    """

    dot = Digraph(comment="GW-Optical ALBEF Framework v5 (final)", format="png")
    dot.attr(rankdir="TB")
    dot.attr(splines="ortho")
    dot.attr(compound="true")
    dot.attr(nodesep="0.28")
    dot.attr(ranksep="0.45")
    dot.attr(bgcolor="white")
    dot.attr(fontname="Helvetica")
    dot.attr(
        label="GW-Optical ALBEF v5\nFinal physical_dual_hgw architecture",
        labelloc="t",
        fontsize="20",
    )
    dot.attr("node", fontname="Helvetica", fontsize="12", margin="0.10,0.06")
    dot.attr("edge", fontname="Helvetica", fontsize="10", color="#566370")

    c_input = "#F3F4F6"
    c_param = "#E5E7EB"
    c_module = "#DCEBFF"
    c_embed = "#D8F3DC"
    c_tensor = "#FFF1CC"
    c_loss = "#F9D5D3"
    c_note = "#EEF2FF"

    input_style = {"shape": "box", "style": "rounded,filled", "fillcolor": c_input, "color": "#B8C0CC"}
    param_style = {"shape": "box", "style": "rounded,filled", "fillcolor": c_param, "color": "#B8C0CC"}
    module_style = {"shape": "box", "style": "rounded,filled", "fillcolor": c_module, "color": "#7AA2D8"}
    embed_style = {"shape": "box", "style": "rounded,filled", "fillcolor": c_embed, "color": "#75B798"}
    tensor_style = {"shape": "ellipse", "style": "filled", "fillcolor": c_tensor, "color": "#D8AE4D"}
    loss_style = {"shape": "hexagon", "style": "filled", "fillcolor": c_loss, "color": "#D27C7C"}
    note_style = {"shape": "note", "style": "filled", "fillcolor": c_note, "color": "#A5B4FC", "fontsize": "10"}

    with dot.subgraph(name="cluster_inputs") as c:
        c.attr(label="Inputs", color="#D1D5DB", style="rounded")
        c.node("GW_Scalar_In", "GW Scalars\nmass, spin, distance", **input_style)
        c.node("GW_Skymap_In", "GW Skymap\n7 x 19200", **input_style)
        c.node("Opt_LC_In", "Optical Light Curves", **input_style)
        c.node("Opt_Coord_In", "Optical Coordinates\nRA, Dec", **input_style)
        c.node("Ref_Time", "Reference Time Grid", **param_style)
        c.node("CLS_Token", "Learnable CLS Token", **param_style)

    with dot.subgraph(name="cluster_gw") as c:
        c.attr(label="GW Encoder (decoupled)", color="#D1D5DB", style="rounded")
        c.node("GW_Scalar_Enc", "GWScalarEncoder", **module_style)
        c.node("GW_Skymap_Enc", "GWSkymapEncoder\n1D ResNet / Lightweight", **module_style)
        c.node("g_scalar", "g_scalar", **tensor_style)
        c.node("H_gw", "H_gw\nGW token sequence", **tensor_style)
        c.node("GW_Pool", "Global Avg Pool", **module_style)
        c.node("g_sky", "g_sky\n= pool(H_gw)", **tensor_style)

    with dot.subgraph(name="cluster_opt") as c:
        c.attr(label="Optical Encoder (decoupled)", color="#D1D5DB", style="rounded")
        c.node("Curve_Enc", "OpticalLightCurveEncoder\nmTAN without coord fusion", **module_style)
        c.node("Coord_Enc", "OpticalCoordEncoder", **embed_style)
        c.node("z_curve", "z_curve", **tensor_style)
        c.node("H_l", "H_l\nOptical token sequence", **tensor_style)
        c.node("coord_feat", "coord_feat", **tensor_style)

    with dot.subgraph(name="cluster_align") as c:
        c.attr(label="Contrastive Branch", color="#93C5FD", style="rounded,dashed")
        c.node("GW_FuseProj", "GWContrastiveFuseProj\nfusion + projection", **module_style)
        c.node("Opt_FuseProj", "OptContrastiveFuseProj\nfusion + projection", **module_style)
        c.node("feat_g", "feat_g\nnormalized", **tensor_style)
        c.node("feat_o", "feat_o\nnormalized", **tensor_style)
        c.node("Pair_Sim", "sim_itc_pair\n(optional for CLS input)", **tensor_style)
        c.node("ITC_Loss", "ITC / SupCon Loss", **loss_style)

    with dot.subgraph(name="cluster_cls") as c:
        c.attr(label="Classification Branch (dual cross-attention)", color="#FCA5A5", style="rounded,dashed")
        c.node("Cross_G2O", "Cross Attention\nGW scalar -> Optical tokens", **module_style)
        c.node("Cross_O2G", "Cross Attention\nCoord -> H_gw tokens", **module_style)
        c.node("Fused_Opt", "f_param2opt", **tensor_style)
        c.node("Fused_GW", "f_coord2gw", **tensor_style)
        c.node("Fusion_Concat", "Concat", **module_style)
        c.node("Classifier", "Classifier MLP", **module_style)
        c.node("CLS_Loss", "Classification Loss", **loss_style)
        c.node(
            "CLS_Note",
            "Classifier inputs:\n[f_param2opt, f_coord2gw]\n+ sim_itc_pair (optional)\n+ cred_level (optional)",
            **note_style,
        )

    dot.edge("GW_Scalar_In", "GW_Scalar_Enc")
    dot.edge("GW_Skymap_In", "GW_Skymap_Enc")
    dot.edge("GW_Scalar_Enc", "g_scalar")
    dot.edge("GW_Skymap_Enc", "H_gw")
    dot.edge("H_gw", "GW_Pool")
    dot.edge("GW_Pool", "g_sky")

    dot.edge("Opt_LC_In", "Curve_Enc")
    dot.edge("Ref_Time", "Curve_Enc")
    dot.edge("CLS_Token", "Curve_Enc")
    dot.edge("Opt_Coord_In", "Coord_Enc")
    dot.edge("Curve_Enc", "z_curve")
    dot.edge("Curve_Enc", "H_l")
    dot.edge("Coord_Enc", "coord_feat")

    dot.edge("g_scalar", "GW_FuseProj")
    dot.edge("g_sky", "GW_FuseProj")
    dot.edge("GW_FuseProj", "feat_g")
    dot.edge("z_curve", "Opt_FuseProj")
    dot.edge("coord_feat", "Opt_FuseProj")
    dot.edge("Opt_FuseProj", "feat_o")
    dot.edge("feat_g", "ITC_Loss")
    dot.edge("feat_o", "ITC_Loss")

    dot.edge("g_scalar", "Cross_G2O", label="Query")
    dot.edge("H_l", "Cross_G2O", label="Key / Value")
    dot.edge("coord_feat", "Cross_O2G", label="Query")
    dot.edge("H_gw", "Cross_O2G", label="Key / Value")
    dot.edge("Cross_G2O", "Fused_Opt")
    dot.edge("Cross_O2G", "Fused_GW")

    dot.edge("feat_g", "Pair_Sim")
    dot.edge("feat_o", "Pair_Sim")

    dot.edge("Fused_Opt", "Fusion_Concat")
    dot.edge("Fused_GW", "Fusion_Concat")
    dot.edge("Pair_Sim", "Fusion_Concat", style="dashed")
    dot.edge("Fusion_Concat", "Classifier")
    dot.edge("Classifier", "CLS_Loss")
    dot.edge("CLS_Note", "Classifier", style="dashed", arrowhead="none", color="#A5B4FC")

    output_path = dot.render(filename, cleanup=False, view=False)
    print(f"Framework v5 base diagram generated: {output_path}")
    return output_path


def create_framework_v3_diagram(
    filename="/fred/oz016/bgao_kn/ML+GW+KN/Model/framework/gw_optical_architecture_v5_clean"
):
    return create_framework_v5_diagram(filename=filename)


if __name__ == "__main__":
    create_framework_v5_diagram()
