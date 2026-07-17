#!/usr/bin/env python3
"""
Plot magnitude vs luptitude as a function of flux (nJy) for each LSST band.

Uses the same luptitude formula as ``scripts/data/create_datasets.py``:
    m_lupt = psfflux_zp - (2.5/ln10) * [arcsinh(f / (2*b)) + ln(b)]
    m_mag  = psfflux_zp - 2.5 * log10(f)      (classical, f > 0 only)

where b = lupt_k * f_5sigma / 5, f_5sigma = 10^((psfflux_zp - m5) / 2.5).
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- parameters (same defaults as scripts/data/create_datasets.py) ----------
PSFFLUX_ZP = 31.4
LUPT_K = 1.0
LUPT_M5_MAG = np.array([23.9, 25.0, 24.7, 24.0, 23.3, 22.1])  # u,g,r,i,z,Y
BANDS = ("u", "g", "r", "i", "z", "Y")
ASINH_MAG_FACTOR = 2.5 / np.log(10.0)
OPTICAL_ONLY_DIR = Path(__file__).resolve().parents[2]
OUTPUT_PNG = OPTICAL_ONLY_DIR / "outputs" / "plots" / "mag_vs_luptitude.png"

# derive b for each band
lupt_f5sigma_njy = 10.0 ** ((PSFFLUX_ZP - LUPT_M5_MAG) / 2.5)
lupt_b_njy = LUPT_K * (lupt_f5sigma_njy / 5.0)

# flux range: show negative-flux region and moderate bright end, in nJy
flux = np.linspace(-1000, 10000, 4000)

# colours for each band
band_colors = {
    "u": "#7b2eff",
    "g": "#00b359",
    "r": "#e63900",
    "i": "#cc7a00",
    "z": "#994d80",
    "Y": "#333333",
}

fig, ax = plt.subplots(figsize=(10, 7))

# Classical magnitude (only valid for f > 0)
flux_pos = flux[flux > 0]
mag = PSFFLUX_ZP - 2.5 * np.log10(flux_pos)

# Luptitude (valid for any flux)
for idx, (band, b_val) in enumerate(zip(BANDS, lupt_b_njy)):
    color = band_colors[band]

    lupt = PSFFLUX_ZP - ASINH_MAG_FACTOR * (
        np.arcsinh(flux / (2.0 * b_val)) + np.log(b_val)
    )
    ax.plot(flux, lupt, color=color, lw=2.0, label=f"LSST-{band}")

    # mark zero-flux level
    lupt_at_zero = PSFFLUX_ZP - ASINH_MAG_FACTOR * (
        np.arcsinh(0.0) + np.log(b_val)
    )
    ax.axhline(lupt_at_zero, color=color, lw=0.4, ls=":", alpha=0.5)

ax.plot(flux_pos, mag, color="gray", lw=1.5, ls="--", label="Classical mag")

ax.axvline(0, color="k", lw=0.5, ls=":")

ax.set_xlabel("Flux (nJy)", fontsize=12)
ax.set_ylabel("Mag / Luptitude (AB)", fontsize=12)
ax.set_title(
    "Magnitude vs Luptitude as a function of flux\n"
    f"(psfflux_zp={PSFFLUX_ZP}, lupt_k={LUPT_K})",
    fontsize=14, fontweight="bold",
)
ax.legend(fontsize=16, loc="lower right")
ax.invert_yaxis()
ax.set_xlim(-1000, 10000)
ax.grid(True, alpha=0.3)
plt.tight_layout()
OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
# plt.savefig(OUTPUT_PNG.with_suffix(".pdf"),
#             bbox_inches="tight")
print("Saved: mag_vs_luptitude.png / .pdf")
