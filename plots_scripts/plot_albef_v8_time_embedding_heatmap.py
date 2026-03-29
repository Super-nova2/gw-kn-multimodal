"""
Plot ALBEF V8 mTAN time-embedding period heatmap (all heads × all dimensions).
Loads parameters directly from the best checkpoint.
"""

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ── paths ──
CKPT_PATH = Path('<BASE_DIR>/data/model/checkpoints_bns_nsbh/bns_nsbh_v8/ALBEF/albef_best.pth')
OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'figures/mtan_optical_encoder_comparison'
OUTPUT_PNG = OUTPUT_DIR / 'albef_v8_mtan_time_embedding_periods_heatmap.png'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── time scale ──
# Matches neg_offset_scale_days_divisor / LearnablePeriodicEmbedding default
TIME_SCALE_DAYS = 100.0

# ── load checkpoint ──
ckpt = torch.load(CKPT_PATH, map_location='cpu')
sd = ckpt['model_state_dict']

# log_wi shape: [1, 1, H, d_r-1]
log_wi = sd['_orig_mod.optical_encoder.curve_encoder.time_embedding.log_wi']  # [1,1,4,63]
log_wi = log_wi.squeeze(0).squeeze(0)  # [H, d_r-1]
num_heads, num_periodic_dims = log_wi.shape

# ── compute periods ──
# omega_i (in scaled-time units) = exp(log_wi)
# period_scaled = 2π / omega_i
# period_days   = period_scaled * TIME_SCALE_DAYS
omega = torch.exp(log_wi).numpy()
period_days = 2.0 * np.pi / omega * TIME_SCALE_DAYS  # [H, d_r-1]

print(f'Checkpoint epoch: {ckpt["epoch"]}')
print(f'Heads: {num_heads},  periodic dims: {num_periodic_dims}')
print(f'Period range — min: {period_days.min():.2f} d, max: {period_days.max():.2f} d')

# ── log10 transform ──
log_heat = np.log10(period_days)  # [H, d_r-1]

# colour limits matching reference figure
vmin, vmax = 0.0, 2.0

# ── plot ──
fig, ax = plt.subplots(figsize=(12.0, 3.2))

im = ax.imshow(log_heat, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)

# x-axis: every 3rd periodic dimension (matching reference style)
xtick_step = 3
xticks = np.arange(0, num_periodic_dims, xtick_step)
ax.set_xticks(xticks)
ax.set_xticklabels([f'dim_{d + 1:03d}' for d in xticks], rotation=45, ha='right', fontsize=8)

# y-axis: heads
ax.set_yticks(np.arange(num_heads))
ax.set_yticklabels([f'head {h}' for h in range(num_heads)])
ax.set_ylabel('attention head')
ax.set_xlabel('periodic embedding dimension')
ax.set_title('ALBEF V8', fontsize=12)

fig.suptitle('mTAN time-embedding periods', fontsize=13, y=1.01)

# colour bar (horizontal, below plot — matching reference)
cb = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.07, pad=0.38)
cb.set_label('log10(period [days])')

fig.savefig(OUTPUT_PNG, dpi=180, bbox_inches='tight')
print(f'Saved: {OUTPUT_PNG}')
plt.close(fig)
