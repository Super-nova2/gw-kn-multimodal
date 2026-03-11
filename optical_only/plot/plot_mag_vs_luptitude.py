#!/usr/bin/env python3
"""
Plot magnitude vs luptitude as a function of flux (nJy) for each LSST band.

Uses the same luptitude formula as create_optical_only_datasets.py:
    m_lupt = psfflux_zp - (2.5/ln10) * [arcsinh(f / (2*b)) + ln(b)]
    m_mag  = psfflux_zp - 2.5 * log10(f)      (classical, f > 0 only)

where b = lupt_k * f_5sigma / 5, f_5sigma = 10^((psfflux_zp - m5) / 2.5).
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------- parameters (same defaults as create_optical_only_datasets.py) ----------
PSFFLUX_ZP = 31.4
LUPT_K = 1.0
LUPT_M5_MAG = np.array([23.9, 25.0, 24.7, 24.0, 23.3, 22.1])  # u,g,r,i,z,Y
BANDS = ("u", "g", "r", "i", "z", "Y")
ASINH_MAG_FACTOR = 2.5 / np.log(10.0)

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

fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharex=False, sharey=False)
axes = axes.ravel()

for idx, (band, b_val) in enumerate(zip(BANDS, lupt_b_njy)):
    ax = axes[idx]

    # Classical magnitude (only valid for f > 0)
    flux_pos = flux[flux > 0]
    mag = PSFFLUX_ZP - 2.5 * np.log10(flux_pos)

    # Luptitude (valid for any flux)
    lupt = PSFFLUX_ZP - ASINH_MAG_FACTOR * (
        np.arcsinh(flux / (2.0 * b_val)) + np.log(b_val)
    )

    ax.plot(flux, lupt, color=band_colors[band], lw=2.0, label="Luptitude")
    ax.plot(flux_pos, mag, color="gray", lw=1.5, ls="--", label="Magnitude")

    # mark zero-flux level
    lupt_at_zero = PSFFLUX_ZP - ASINH_MAG_FACTOR * (
        np.arcsinh(0.0) + np.log(b_val)
    )
    ax.axvline(0, color="k", lw=0.5, ls=":")
    ax.axhline(lupt_at_zero, color=band_colors[band], lw=0.5, ls=":", alpha=0.6)

    # mark 5-sigma flux
    ax.axvline(lupt_f5sigma_njy[idx], color=band_colors[band], lw=0.7, ls="-.",
               alpha=0.5, label=f"$f_{{5\\sigma}}$={lupt_f5sigma_njy[idx]:.0f} nJy")

    ax.set_title(f"LSST-{band}  ($m_5$={LUPT_M5_MAG[idx]}, $b$={b_val:.1f} nJy)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Flux (nJy)", fontsize=11)
    ax.set_ylabel("Mag / Luptitude (AB)", fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    ax.invert_yaxis()
    ax.set_xlim(-1000, 10000)
    ax.grid(True, alpha=0.3)

fig.suptitle(
    "Magnitude vs Luptitude as a function of flux\n"
    f"(psfflux_zp={PSFFLUX_ZP}, lupt_k={LUPT_K})",
    fontsize=14, fontweight="bold", y=0.98,
)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig("/fred/oz016/bgao_kn/ML+GW+KN/optical_only/mag_vs_luptitude.png",
            dpi=300, bbox_inches="tight")
# plt.savefig("/fred/oz016/bgao_kn/ML+GW+KN/optical_only/mag_vs_luptitude.pdf",
#             bbox_inches="tight")
print("Saved: mag_vs_luptitude.png / .pdf")
