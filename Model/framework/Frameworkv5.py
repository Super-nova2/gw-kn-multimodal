from pathlib import Path

from graphviz import Digraph


DEFAULT_V5_BASE = (
    "/fred/oz016/bgao_kn/ML+GW+KN/Model/framework/gw_optical_architecture_v5_clean"
)


def _normalize_output_base(filename):
    path = Path(filename)
    if path.suffix in {".dot", ".png", ".svg"}:
        return path.with_suffix("")
    return path


def build_framework_v5_diagram():
    """Build the v5 architecture diagram using the clean v4 visual style."""

    dot = Digraph("G", comment="GW-Optical ALBEF architecture (v5 clean, v4 style)")
    dot.attr(rankdir="LR", labelloc="t", label="GW-Optical ALBEF Architecture (Clean v5)")
    dot.attr(
        "graph",
        fontname="Helvetica",
        fontsize="18",
        bgcolor="#FFFFFF",
        pad="0.15",
        nodesep="0.45",
        ranksep="0.75",
        splines="spline",
        dpi="220",
    )
    dot.attr(
        "node",
        shape="box",
        style="rounded,filled",
        fontname="Helvetica",
        fontsize="11",
        color="#475569",
        penwidth="1.2",
    )
    dot.attr("edge", color="#64748B", penwidth="1.2", arrowsize="0.75")

    input_style = {"fillcolor": "#F8FAFC"}
    gw_style = {"fillcolor": "#DBEAFE", "color": "#60A5FA"}
    opt_style = {"fillcolor": "#DCFCE7", "color": "#22C55E"}
    align_style = {"fillcolor": "#E0E7FF", "color": "#6366F1"}
    fusion_style = {"fillcolor": "#FEF3C7", "color": "#F59E0B"}
    tensor_style = {
        "shape": "ellipse",
        "fillcolor": "#FFF7ED",
        "color": "#FB923C",
    }
    loss_style = {
        "shape": "diamond",
        "style": "filled",
        "fillcolor": "#FEE2E2",
        "color": "#EF4444",
        "fontcolor": "#991B1B",
    }

    with dot.subgraph(name="cluster_inputs") as c:
        c.attr(label="1) Inputs", color="#CBD5E1", style="rounded,dashed")
        c.node("gw_scalars", "GW scalars\n(mass, spin, distance)", **input_style)
        c.node("gw_skymap", "GW skymap sequence\n(7 x 19200)", **input_style)
        c.node("opt_lc", "Optical light curves", **input_style)
        c.node("opt_coord", "Optical coordinates\n(RA, Dec)", **input_style)
        c.node("ref_time", "Reference time grid", **input_style)
        c.node("cls_tok", "Learnable CLS token", **input_style)

    with dot.subgraph(name="cluster_encoders") as c:
        c.attr(label="2) Decoupled Encoders", color="#93C5FD", style="rounded,dashed")
        c.node("gw_scalar_enc", "GW scalar encoder", **gw_style)
        c.node("gw_skymap_enc", "GW skymap encoder\n(1D ResNet / lightweight)", **gw_style)
        c.node("gw_pool", "Global average pool", **gw_style)
        c.node("curve_enc", "Optical light-curve encoder\n(mTAN / Transformer)", **opt_style)
        c.node("coord_enc", "Optical coordinate encoder", **opt_style)

        c.node("g_scalar", "g_scalar\nGW scalar vector", **tensor_style)
        c.node("H_gw", "H_gw\nGW token sequence", **tensor_style)
        c.node("g_sky", "g_sky\nPooled skymap vector", **tensor_style)
        c.node("z_curve", "z_curve\nLight-curve global vector", **tensor_style)
        c.node("H_l", "H_l\nOptical token sequence", **tensor_style)
        c.node("coord_feat", "coord_feat\nCoordinate vector", **tensor_style)

    with dot.subgraph(name="cluster_align") as c:
        c.attr(label="3) Alignment Branch (Contrastive)", color="#A5B4FC", style="rounded,dashed")
        c.node("gw_proj", "GW fusion + projection", **align_style)
        c.node("opt_proj", "Optical fusion + projection", **align_style)
        c.node("feat_g", "feat_g\nNormalized embedding", **tensor_style)
        c.node("feat_o", "feat_o\nNormalized embedding", **tensor_style)
        c.node("pair_sim", "sim_itc_pair\nOptional scalar", **tensor_style)
        c.node("loss_itc", "L_itc", **loss_style)

    with dot.subgraph(name="cluster_fusion") as c:
        c.attr(label="4) Fusion + Classification", color="#FCA5A5", style="rounded,dashed")
        c.node("cross_g2o", "Cross-attention\n(g_scalar -> H_l)", **fusion_style)
        c.node("cross_o2g", "Cross-attention\n(coord_feat -> H_gw)", **fusion_style)
        c.node("cred", "Credible level estimator\n(skymap + RA/Dec)", **fusion_style)
        c.node("f_opt", "f_param2opt", **tensor_style)
        c.node("f_gw", "f_coord2gw", **tensor_style)
        c.node("merge", "Feature concat", **fusion_style)
        c.node("clf", "Classifier MLP", **fusion_style)
        c.node("loss_cls", "L_cls", **loss_style)

    dot.edge("gw_scalars", "gw_scalar_enc")
    dot.edge("gw_skymap", "gw_skymap_enc")
    dot.edge("opt_lc", "curve_enc")
    dot.edge("ref_time", "curve_enc")
    dot.edge("cls_tok", "curve_enc")
    dot.edge("opt_coord", "coord_enc")

    dot.edge("gw_scalar_enc", "g_scalar")
    dot.edge("gw_skymap_enc", "H_gw")
    dot.edge("H_gw", "gw_pool")
    dot.edge("gw_pool", "g_sky")
    dot.edge("curve_enc", "z_curve")
    dot.edge("curve_enc", "H_l")
    dot.edge("coord_enc", "coord_feat")

    dot.edge("g_scalar", "gw_proj")
    dot.edge("g_sky", "gw_proj")
    dot.edge("gw_proj", "feat_g")
    dot.edge("z_curve", "opt_proj")
    dot.edge("coord_feat", "opt_proj")
    dot.edge("opt_proj", "feat_o")
    dot.edge("feat_g", "loss_itc")
    dot.edge("feat_o", "loss_itc")
    dot.edge("feat_g", "pair_sim")
    dot.edge("feat_o", "pair_sim")

    dot.edge("g_scalar", "cross_g2o")
    dot.edge("H_l", "cross_g2o")
    dot.edge("coord_feat", "cross_o2g")
    dot.edge("H_gw", "cross_o2g")
    dot.edge("cross_g2o", "f_opt")
    dot.edge("cross_o2g", "f_gw")

    # dot.edge("gw_skymap", "cred")
    # dot.edge("opt_coord", "cred")

    dot.edge("f_opt", "merge")
    dot.edge("f_gw", "merge")
    dot.edge("pair_sim", "merge", style="dashed", label="optional")
    dot.edge("cred", "merge", style="dashed", label="optional")
    dot.edge("merge", "clf")
    dot.edge("clf", "loss_cls")

    return dot


def create_framework_v5_diagram(
    filename=DEFAULT_V5_BASE
):
    base_path = _normalize_output_base(filename)
    dot = build_framework_v5_diagram()

    dot_path = base_path.with_suffix(".dot")
    dot.save(dot_path)

    png_path = dot.render(str(base_path), format="png", cleanup=True, view=False)
    svg_path = dot.render(str(base_path), format="svg", cleanup=True, view=False)

    print(f"Framework v5 diagram generated: {png_path}")
    print(f"DOT source saved: {dot_path}")
    print(f"SVG saved: {svg_path}")
    return {"dot": str(dot_path), "png": png_path, "svg": svg_path}


def create_framework_v3_diagram(
    filename=DEFAULT_V5_BASE
):
    return create_framework_v5_diagram(filename=filename)


if __name__ == "__main__":
    create_framework_v5_diagram()
