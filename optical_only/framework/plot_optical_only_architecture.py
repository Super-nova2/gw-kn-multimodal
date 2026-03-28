"""
绘制 Optical-Only KN 分类模型架构图（含对抗训练部分）。

输出：
  optical_only_architecture_adv.png  (300 dpi)
  optical_only_architecture_adv.pdf

用法：
  python plot_optical_only_architecture.py
"""

from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── 输出目录 ──────────────────────────────────────────────────────────────────
OUTDIR = Path(__file__).parent
OUTDIR.mkdir(parents=True, exist_ok=True)

# ── 配色 ─────────────────────────────────────────────────────────────────────
C = {
    "input":      "#DBEAFE",   # 蓝
    "input_bd":   "#2563EB",
    "embed":      "#D1FAE5",   # 绿
    "embed_bd":   "#059669",
    "attn":       "#EDE9FE",   # 紫
    "attn_bd":    "#7C3AED",
    "pool":       "#F3E8FF",   # 淡紫
    "pool_bd":    "#9333EA",
    "cls":        "#FEF9C3",   # 黄
    "cls_bd":     "#CA8A04",
    "head":       "#FCE7F3",   # 粉
    "head_bd":    "#DB2777",
    "output":     "#FEE2E2",   # 红
    "output_bd":  "#DC2626",
    "proj":       "#CCFBF1",   # 青
    "proj_bd":    "#0D9488",
    "grl":        "#FEF3C7",   # 橙
    "grl_bd":     "#D97706",
    "adv":        "#FDE8D8",   # 浅橙
    "adv_bd":     "#EA580C",
    "loss":       "#F1F5F9",   # 灰
    "loss_bd":    "#64748B",
    "arrow":      "#475569",
    "arrow_adv":  "#EA580C",
    "arrow_grl":  "#B45309",
    "bg":         "#F8FAFC",
}

import matplotlib.font_manager as _fm
_fm.fontManager.addfont("/usr/share/fonts/google-droid-sans-fonts/DroidSansFallbackFull.ttf")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Droid Sans", "Droid Sans Fallback", "DejaVu Sans"]
FONT = "sans-serif"


def box(ax, x, y, w, h, label, fc, ec, fontsize=9, alpha=1.0, linestyle="-",
        linewidth=1.4, bold=False, sublabel=None):
    """绘制圆角矩形 + 文字"""
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02",
        facecolor=fc, edgecolor=ec,
        linewidth=linewidth, linestyle=linestyle,
        alpha=alpha, zorder=3,
    )
    ax.add_patch(patch)
    if sublabel:
        ax.text(x, y + 0.018, label, ha="center", va="center",
                fontsize=fontsize, fontfamily=FONT,
                fontweight="bold" if bold else "normal", zorder=4)
        ax.text(x, y - 0.022, sublabel, ha="center", va="center",
                fontsize=fontsize - 1.2, fontfamily=FONT,
                color="#4B5563", zorder=4)
    else:
        ax.text(x, y, label, ha="center", va="center",
                fontsize=fontsize, fontfamily=FONT,
                fontweight="bold" if bold else "normal", zorder=4,
                multialignment="center")


def arrow(ax, x0, y0, x1, y1, color="#475569", lw=1.3,
          style="->", ls="-", rad=0.0, shrink=4):
    ax.annotate("",
        xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle=style,
            color=color,
            lw=lw,
            linestyle=ls,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=shrink, shrinkB=shrink,
        ),
        zorder=5,
    )


def bracket_label(ax, x, y1, y2, text, color="#64748B", fontsize=8):
    """右侧大括号标注"""
    ax.annotate("", xy=(x, y1), xytext=(x, y2),
                arrowprops=dict(arrowstyle="-", color=color, lw=1.0,
                                connectionstyle="arc3,rad=0"))
    ax.text(x + 0.015, (y1 + y2) / 2, text, ha="left", va="center",
            fontsize=fontsize, color=color, fontfamily=FONT,
            fontstyle="italic")


# ─────────────────────────────────────────────────────────────────────────────
# 画布
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 1.0)
ax.set_ylim(0, 1.0)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor(C["bg"])
ax.set_facecolor(C["bg"])

# ─────────────────────────────────────────────────────────────────────────────
# 主干（从左到右，y=0.62）
# ─────────────────────────────────────────────────────────────────────────────
Y_MAIN = 0.62
BH = 0.10   # box height
BW_S = 0.10  # small box width
BW_M = 0.12  # medium
BW_L = 0.14  # large

positions = {}   # name -> (x, y, w, h)

def add(name, x, y, w, h, label, fc, ec, fs=8.5, bold=False):
    box(ax, x, y, w, h, label, fc, ec, fontsize=fs, bold=bold)
    positions[name] = (x, y, w, h)

# ── 输入区 ────────────────────────────────────────────────────────────────────
add("input", 0.07, Y_MAIN, 0.11, BH,
    "观测序列输入\n"
    r"$(\mathbf{t},\mathbf{v},\mathbf{m},\boldsymbol{\sigma})$"
    "\n[B, L, 6 bands]",
    C["input"], C["input_bd"], fs=8)

add("reftime", 0.07, Y_MAIN - 0.20, 0.11, 0.08,
    "参考时刻网格\n"
    r"$\mathbf{t}_{ref}$ [B, 64]",
    C["input"], C["input_bd"], fs=8)

# ── 时间嵌入 ──────────────────────────────────────────────────────────────────
add("embed", 0.225, Y_MAIN, 0.12, BH,
    "周期性时间嵌入\n"
    r"$\phi_h(t)=\{w_0 t,\sin(w_i t+a_i)\}$"
    "\n[B, L/64, H=4, d=64]",
    C["embed"], C["embed_bd"], fs=7.8)

# ── mTAN 注意力 ───────────────────────────────────────────────────────────────
add("mtan", 0.395, Y_MAIN, 0.13, BH,
    "多头时间注意力\n(mTAN)\n"
    "掩码 + SNR 加权\n[B, 65, J=128]",
    C["attn"], C["attn_bd"], fs=8)

# ── CLS / 时序特征分支 ────────────────────────────────────────────────────────
Y_CLS  = Y_MAIN + 0.18
Y_TEMP = Y_MAIN - 0.18

add("cls_feat", 0.545, Y_CLS, 0.11, 0.08,
    "CLS 向量\n"
    r"$\mathbf{z} \in \mathbb{R}^{B\times 128}$",
    C["cls"], C["cls_bd"], fs=8)

add("temp_feat", 0.545, Y_TEMP, 0.11, 0.08,
    "时序特征\n"
    r"$H \in \mathbb{R}^{B\times 64\times 128}$",
    C["cls"], C["cls_bd"], fs=8)

add("pool", 0.545, Y_MAIN, 0.11, 0.07,
    "时序均值池化\n"
    r"$\bar{\mathbf{h}} \in \mathbb{R}^{B\times 128}$",
    C["pool"], C["pool_bd"], fs=8)

# ── 特征拼接 ──────────────────────────────────────────────────────────────────
add("concat", 0.680, Y_MAIN, 0.10, BH,
    "特征拼接\n"
    r"$[\mathbf{z};\bar{\mathbf{h}}]$"
    "\n[B, 256]",
    C["pool"], C["pool_bd"], fs=8)

# ── 分类头 ────────────────────────────────────────────────────────────────────
add("cls_head", 0.800, Y_MAIN, 0.10, BH,
    "分类头 (MLP)\n"
    "256→128→1\nReLU + Dropout",
    C["head"], C["head_bd"], fs=8)

# ── 输出 ──────────────────────────────────────────────────────────────────────
add("output", 0.920, Y_MAIN, 0.09, BH,
    "KN 概率\n"
    r"$p = \sigma(\hat{y})$",
    C["output"], C["output_bd"], fs=8.5, bold=True)

# ─────────────────────────────────────────────────────────────────────────────
# 对抗训练支路（从 concat 向下）
# ─────────────────────────────────────────────────────────────────────────────
Y_GRL   = 0.22
Y_ADV   = 0.10

add("grl", 0.680, Y_GRL, 0.10, 0.07,
    "梯度反转层 (GRL)\n"
    r"$\lambda_{\mathrm{GRL}}=1.0$",
    C["grl"], C["grl_bd"], fs=8)

# 三个对抗头横排
X_ADV = [0.530, 0.680, 0.830]
ADV_LABELS = [
    "对抗头：探测数\n"
    r"$N_{\mathrm{det}}$ (5 类)",
    "对抗头：波段数\n"
    r"$N_{\mathrm{bands}}$ (4 类)",
    "对抗头：时间跨度\n"
    r"$\Delta t$ (5 类)",
]
for xi, lbl in zip(X_ADV, ADV_LABELS):
    box(ax, xi, Y_ADV, 0.12, 0.07, lbl, C["adv"], C["adv_bd"], fontsize=7.8)
    positions[f"adv_{xi}"] = (xi, Y_ADV, 0.12, 0.07)

# 投影头（用于一致性损失，从 concat 向上偏右）
add("proj", 0.800, 0.88, 0.10, 0.07,
    "投影头 (MLP)\n"
    r"256→128→64, $\ell_2$ 归一化",
    C["proj"], C["proj_bd"], fs=7.8)

# ─────────────────────────────────────────────────────────────────────────────
# 损失标注区（右侧）
# ─────────────────────────────────────────────────────────────────────────────
Y_LOSS_CLS  = Y_MAIN
Y_LOSS_CONS = 0.88
Y_LOSS_ADV  = Y_ADV

def loss_label(ax, x, y, text, color):
    ax.text(x, y, text, ha="left", va="center", fontsize=8,
            fontfamily=FONT, color=color,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color,
                      lw=0.8, alpha=0.85),
            zorder=6)

loss_label(ax, 0.965, Y_LOSS_CLS,
           r"$\mathcal{L}_{\mathrm{cls}}$: BCE", C["head_bd"])
loss_label(ax, 0.965, Y_LOSS_CONS,
           r"$\mathcal{L}_{\mathrm{cons}}$: 嵌入余弦 + 对称KL", C["proj_bd"])
loss_label(ax, 0.965, Y_LOSS_ADV,
           r"$\mathcal{L}_{\mathrm{adv}}$: CE × 3（经GRL反传）", C["adv_bd"])

# ─────────────────────────────────────────────────────────────────────────────
# 箭头：主干
# ─────────────────────────────────────────────────────────────────────────────
def mid_right(name): p=positions[name]; return (p[0]+p[2]/2, p[1])
def mid_left(name):  p=positions[name]; return (p[0]-p[2]/2, p[1])
def mid_top(name):   p=positions[name]; return (p[0], p[1]+p[3]/2)
def mid_bot(name):   p=positions[name]; return (p[0], p[1]-p[3]/2)

# input -> embed
arrow(ax, *mid_right("input"), *mid_left("embed"), C["arrow"])
# reftime -> embed (斜向)
arrow(ax, *mid_right("reftime"),
      positions["embed"][0] - positions["embed"][2]/2,
      Y_MAIN - 0.08, C["input_bd"], lw=1.0, rad=0.15)
# embed -> mtan
arrow(ax, *mid_right("embed"), *mid_left("mtan"), C["arrow"])
# mtan -> cls_feat (向上)
arrow(ax,
      positions["mtan"][0], positions["mtan"][1] + positions["mtan"][3]/2,
      positions["cls_feat"][0], positions["cls_feat"][1] - positions["cls_feat"][3]/2,
      C["arrow"])
# mtan -> temp_feat (向下)
arrow(ax,
      positions["mtan"][0], positions["mtan"][1] - positions["mtan"][3]/2,
      positions["temp_feat"][0], positions["temp_feat"][1] + positions["temp_feat"][3]/2,
      C["arrow"])
# cls_feat -> concat (右斜下)
arrow(ax,
      positions["cls_feat"][0] + positions["cls_feat"][2]/2, positions["cls_feat"][1],
      positions["concat"][0] - positions["concat"][2]/2, positions["concat"][1] + 0.025,
      C["arrow"])
# temp_feat -> pool
arrow(ax,
      positions["temp_feat"][0] + positions["temp_feat"][2]/2, positions["temp_feat"][1],
      positions["pool"][0] - positions["pool"][2]/2, positions["pool"][1],
      C["arrow"])
# pool -> concat
arrow(ax, *mid_right("pool"), *mid_left("concat"), C["arrow"])
# concat -> cls_head
arrow(ax, *mid_right("concat"), *mid_left("cls_head"), C["arrow"])
# cls_head -> output
arrow(ax, *mid_right("cls_head"), *mid_left("output"), C["arrow"])

# ── 对抗支路箭头 ──────────────────────────────────────────────────────────────
# concat -> grl (向下)
arrow(ax, *mid_bot("concat"), *mid_top("grl"), C["arrow_grl"], lw=1.2)
# grl -> 三个 adv head
for xi in X_ADV:
    arrow(ax,
          positions["grl"][0], positions["grl"][1] - positions["grl"][3]/2,
          xi, Y_ADV + 0.035,
          C["arrow_adv"], lw=1.1, rad=0.0 if xi == 0.680 else (0.15 if xi < 0.680 else -0.15))

# concat -> proj (向右上)
arrow(ax,
      positions["concat"][0] + positions["concat"][2]/2, positions["concat"][1],
      positions["proj"][0] - positions["proj"][2]/2, positions["proj"][1],
      C["proj_bd"], lw=1.1, rad=-0.25)

# proj -> loss_cons (虚线标注方向)
arrow(ax,
      positions["proj"][0] + positions["proj"][2]/2, positions["proj"][1],
      0.960, Y_LOSS_CONS,
      C["proj_bd"], lw=0.9, ls="dashed", style="-|>")

# output -> loss_cls
arrow(ax,
      positions["output"][0] + positions["output"][2]/2, positions["output"][1],
      0.960, Y_LOSS_CLS,
      C["head_bd"], lw=0.9, ls="dashed", style="-|>")

# adv 中间头 -> loss_adv
arrow(ax,
      X_ADV[1] + 0.06, Y_ADV,
      0.960, Y_LOSS_ADV,
      C["adv_bd"], lw=0.9, ls="dashed", style="-|>")

# ─────────────────────────────────────────────────────────────────────────────
# 阶段标注框
# ─────────────────────────────────────────────────────────────────────────────
def stage_rect(ax, x0, y0, x1, y1, label, color, fontsize=7.5):
    rect = mpatches.FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle="round,pad=0.01",
        facecolor="none", edgecolor=color,
        linewidth=1.2, linestyle="--", zorder=2, alpha=0.7,
    )
    ax.add_patch(rect)
    ax.text((x0 + x1) / 2, y1 + 0.012, label,
            ha="center", va="bottom", fontsize=fontsize,
            color=color, fontfamily=FONT, fontweight="bold", zorder=6)

# 阶段 1：仅分类头
stage_rect(ax, 0.740, Y_MAIN - 0.065, 0.870, Y_MAIN + 0.065,
           "阶段 1：仅训练分类头", C["head_bd"])

# 阶段 2：一致性
stage_rect(ax, 0.620, 0.810, 0.960, 0.940,
           "阶段 2：+ 一致性损失", C["proj_bd"])

# 阶段 3：对抗
stage_rect(ax, 0.460, 0.040, 0.960, 0.295,
           "阶段 3：+ 梯度反转对抗训练", C["adv_bd"])

# ─────────────────────────────────────────────────────────────────────────────
# GRL 反向标注
# ─────────────────────────────────────────────────────────────────────────────
ax.annotate(
    "反向传播：梯度取反",
    xy=(0.680, Y_GRL - 0.005),
    xytext=(0.560, Y_GRL - 0.06),
    fontsize=7.5, color=C["grl_bd"], fontfamily=FONT,
    arrowprops=dict(arrowstyle="->", color=C["grl_bd"], lw=0.9),
    zorder=7,
)

# ─────────────────────────────────────────────────────────────────────────────
# 总损失公式
# ─────────────────────────────────────────────────────────────────────────────
ax.text(0.50, 0.025,
        r"总损失（阶段3）：$\mathcal{L} = "
        r"\mathcal{L}_{\mathrm{cls}} "
        r"+ \lambda_{\mathrm{emb}}\mathcal{L}_{\mathrm{emb}} "
        r"+ \lambda_{\mathrm{prob}}\mathcal{L}_{\mathrm{prob}} "
        r"+ \lambda_{\mathrm{det}}\mathcal{L}_{\mathrm{det}} "
        r"+ \lambda_{\mathrm{band}}\mathcal{L}_{\mathrm{band}} "
        r"+ \lambda_{\mathrm{span}}\mathcal{L}_{\mathrm{span}}$",
        ha="center", va="center", fontsize=8.5, fontfamily=FONT,
        color="#1E293B",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#94A3B8", lw=0.8),
        zorder=7)

# ─────────────────────────────────────────────────────────────────────────────
# 标题
# ─────────────────────────────────────────────────────────────────────────────
ax.set_title("Optical-Only KN 分类模型架构（含对抗训练）",
             fontsize=13, fontfamily=FONT, fontweight="bold",
             color="#1E293B", pad=8)

# ─────────────────────────────────────────────────────────────────────────────
# 图例
# ─────────────────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(fc=C["input"],  ec=C["input_bd"],  label="输入"),
    mpatches.Patch(fc=C["embed"],  ec=C["embed_bd"],  label="时间嵌入"),
    mpatches.Patch(fc=C["attn"],   ec=C["attn_bd"],   label="注意力编码器"),
    mpatches.Patch(fc=C["pool"],   ec=C["pool_bd"],   label="池化 / 融合"),
    mpatches.Patch(fc=C["head"],   ec=C["head_bd"],   label="分类头"),
    mpatches.Patch(fc=C["proj"],   ec=C["proj_bd"],   label="投影头（一致性）"),
    mpatches.Patch(fc=C["grl"],    ec=C["grl_bd"],    label="梯度反转层"),
    mpatches.Patch(fc=C["adv"],    ec=C["adv_bd"],    label="对抗辅助头"),
]
ax.legend(handles=legend_items, loc="lower left",
          fontsize=7.5, framealpha=0.9, ncol=4,
          bbox_to_anchor=(0.0, 0.0),
          handlelength=1.2, handleheight=0.9)

# ─────────────────────────────────────────────────────────────────────────────
# 保存
# ─────────────────────────────────────────────────────────────────────────────
plt.tight_layout()
for suffix in ("png", "pdf"):
    out = OUTDIR / f"optical_only_architecture_adv.{suffix}"
    fig.savefig(out, dpi=300, bbox_inches="tight",
                facecolor=C["bg"])
    print(f"已保存：{out}")

plt.close(fig)
