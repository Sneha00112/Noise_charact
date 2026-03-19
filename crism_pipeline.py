"""
CRISM Hyperspectral Dataset - Complete Preprocessing Pipeline
Produces:
  1. EDA visualizations (variance, histogram, spectra, spatial map)
  2. Noise profile bar chart (raw baseline)
  3. Before vs After physical corrections bar chart
  4. Detailed report
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, uniform_filter1d

OUT_DIR = "/Users/snehasr/Desktop/DRDO/Orig CRISMD2/Pipeline_Results"
IMG_PATH = "/Users/snehasr/Desktop/DRDO/Orig CRISMD2/frt0001073b_01_ra156s_trr3.img"
LBL_PATH = "/Users/snehasr/Desktop/DRDO/Orig CRISMD2/frt0001073b_01_ra156s_trr3.lbl.txt"
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# STEP 1: LOAD DATA
# ─────────────────────────────────────────────────────────────
print("=" * 55)
print("STEP 1: LOADING DATASET")
print("=" * 55)

LINES, SAMPLES, BANDS = 15, 64, 107
WAVELENGTHS = np.linspace(1.002, 3.920, BANDS)  # µm, CRISM S-sensor

raw_flat = np.fromfile(IMG_PATH, dtype=np.float32, count=LINES * SAMPLES * BANDS)
cube_raw = raw_flat.reshape((LINES, BANDS, SAMPLES)).transpose(0, 2, 1)  # (L, S, B)
cube_raw = np.nan_to_num(cube_raw, nan=0.0, posinf=0.0, neginf=0.0)

print(f"  Spatial Dimensions : {LINES} Lines x {SAMPLES} Samples")
print(f"  Spectral Bands     : {BANDS}")
print(f"  Wavelength Range   : {WAVELENGTHS[0]:.3f} µm – {WAVELENGTHS[-1]:.3f} µm")
print(f"  Data Type          : float32 (PC_REAL, Little-Endian)")
print(f"  Min / Max          : {cube_raw.min():.4f} / {cube_raw.max():.4f}")
print(f"  Mean / Std         : {cube_raw.mean():.4f} / {cube_raw.std():.4f}")

# Normalise to [0,1] once for consistent metric computation
def norm01(c):
    lo, hi = c.min(), c.max()
    return (c - lo) / (hi - lo + 1e-10)

cube_norm = norm01(cube_raw)

# ─────────────────────────────────────────────────────────────
# STEP 2: EDA – visualisations
# ─────────────────────────────────────────────────────────────
print("\nSTEP 2: EXPLORATORY DATA ANALYSIS")

# 2a  Band-wise variance
bvar = cube_norm.var(axis=(0, 1))
plt.figure(figsize=(10, 4))
plt.plot(WAVELENGTHS, bvar, color='steelblue')
plt.fill_between(WAVELENGTHS, bvar, alpha=0.25, color='steelblue')
plt.xlabel('Wavelength (µm)')
plt.ylabel('Variance')
plt.title('Band-wise Variance  (spikes → noisy bands)')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/eda_01_variance.png", dpi=150)
plt.close()
print("  [saved] eda_01_variance.png")

# 2b  Histogram
plt.figure(figsize=(8, 4))
plt.hist(cube_norm.flatten(), bins=100, color='slategray', log=True, edgecolor='none')
plt.xlabel('Normalised Reflectance')
plt.ylabel('Count  (log)')
plt.title('Reflectance Histogram  (outlier spikes visible at extremes)')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/eda_02_histogram.png", dpi=150)
plt.close()
print("  [saved] eda_02_histogram.png")

# 2c  Spectral profiles
plt.figure(figsize=(10, 5))
rng = np.random.default_rng(42)
for _ in range(5):
    r, c_ = rng.integers(0, LINES), rng.integers(0, SAMPLES)
    plt.plot(WAVELENGTHS, cube_norm[r, c_, :], alpha=0.7, linewidth=0.9)
plt.xlabel('Wavelength (µm)')
plt.ylabel('Normalised Reflectance')
plt.title('Raw Spectral Profiles  (atmospheric dips & random noise visible)')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/eda_03_spectral_profiles.png", dpi=150)
plt.close()
print("  [saved] eda_03_spectral_profiles.png")

# 2d  Spatial map (single band, shows striping)
band_idx = 50
plt.figure(figsize=(7, 4))
plt.imshow(cube_norm[:, :, band_idx], cmap='inferno', aspect='auto')
plt.colorbar(label='Normalised Reflectance')
plt.title(f'Band {band_idx} Spatial Map ({WAVELENGTHS[band_idx]:.2f} µm)  – striping visible')
plt.xlabel('Sample (column)')
plt.ylabel('Line (row)')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/eda_04_spatial_map.png", dpi=150)
plt.close()
print("  [saved] eda_04_spatial_map.png")

# ─────────────────────────────────────────────────────────────
# NOISE METRIC FUNCTIONS  (all return a scalar ≥ 0; lower = less noise)
# ─────────────────────────────────────────────────────────────
def metric_striping(c):
    """Std-dev of column means across bands – captures vertical banding."""
    col_means = c.mean(axis=(0, 2))          # (SAMPLES,)
    return float(col_means.std())

def metric_random_noise(c):
    """Mean per-band spatial std – captures pixel-to-pixel jitter."""
    return float(c.std(axis=(0, 1)).mean())

def metric_spike(c):
    """Fraction of pixels > 3σ from global mean."""
    mu, sigma = c.mean(), c.std()
    return float(np.sum(np.abs(c - mu) > 3 * sigma) / c.size)

def metric_atm_absorption(c):
    """Std of first differences of the global mean spectrum – spectral roughness."""
    spec = c.mean(axis=(0, 1))
    return float(np.diff(spec).std())

def metric_dust(c):
    """Absolute slope of first 15 bands of mean spectrum (dust scattering adds slope)."""
    spec = c.mean(axis=(0, 1))
    coeffs = np.polyfit(np.arange(15), spec[:15], 1)
    return float(abs(coeffs[0]))

def metric_low_snr(c):
    """Mean of 1/(SNR+1) per band – captures bands with poor signal quality."""
    bm = c.mean(axis=(0, 1))
    bs = c.std(axis=(0, 1))
    snr = np.where(bs > 1e-10, bm / bs, 0.0)
    return float((1.0 / (snr + 1.0)).mean())

NOISE_LABELS = [
    'Striping',
    'Random Noise',
    'Spike Noise',
    'Atmospheric\nAbsorption',
    'Dust Scattering',
    'Low SNR Bands',
]

def compute_all(c):
    c = norm01(c)
    return np.array([
        metric_striping(c),
        metric_random_noise(c),
        metric_spike(c),
        metric_atm_absorption(c),
        metric_dust(c),
        metric_low_snr(c),
    ])

raw_metrics = compute_all(cube_norm)

# ─────────────────────────────────────────────────────────────
# CHART 1 – RAW NOISE PROFILE
# ─────────────────────────────────────────────────────────────
def bar_chart(ax, values, labels, color, title):
    y = np.arange(len(labels))
    bars = ax.barh(y, values, color=color, edgecolor='white', linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Noise Intensity (raw metric value, lower = better)')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.xaxis.grid(True, linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)
    for bar, v in zip(bars, values):
        ax.text(v + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f'{v:.4f}', va='center', fontsize=8)

fig, ax = plt.subplots(figsize=(10, 5))
bar_chart(ax, raw_metrics, NOISE_LABELS, '#e05c5c',
          'Noise Profile – Raw CRISM Data (Before Any Corrections)')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/chart1_raw_noise_profile.png", dpi=150)
plt.close()
print("  [saved] chart1_raw_noise_profile.png")

# ─────────────────────────────────────────────────────────────
# STEP 3: PHYSICAL CORRECTIONS
# ─────────────────────────────────────────────────────────────
print("\nSTEP 3: PHYSICAL CORRECTIONS")
cube_work = cube_norm.copy()

# 3a  Spike removal via spatial median filter
#     Targets: Spike Noise, Random Noise
print("  3a  Median spatial filter (spike removal)…")
cube_wk = np.stack(
    [median_filter(cube_work[:, :, b], size=3) for b in range(BANDS)],
    axis=2
)
cube_work = cube_wk

# 3b  Radiometric normalization (per-band min-max)
#     Targets: Random Noise (baseline shifts between bands)
print("  3b  Radiometric normalisation (per-band min-max)…")
b_min = cube_work.min(axis=(0, 1), keepdims=True)
b_max = cube_work.max(axis=(0, 1), keepdims=True)
cube_work = (cube_work - b_min) / (b_max - b_min + 1e-10)

# 3c  Illumination / destriping  (column flat-field)
#     Targets: Striping, Dust (spatial illumination gradient)
print("  3c  Column flat-field destriping…")
col_mean = cube_work.mean(axis=0, keepdims=True)          # (1, S, B)
col_mean[col_mean < 1e-10] = 1e-10
cube_work = cube_work / col_mean
cube_work = norm01(cube_work)

# 3d  Spectral smoothing  (Gaussian along wavelength axis)
#     Targets: Atmospheric absorption rough edges, Dust spectral slope
print("  3d  Uniform spectral smoothing (atmospheric baseline)…")
cube_work = uniform_filter1d(cube_work, size=5, axis=2)
cube_work = norm01(cube_work)

pc_metrics = compute_all(cube_work)

print("\n  Noise metric comparison (raw  →  after physical corrections):")
for lbl, r, p in zip(NOISE_LABELS, raw_metrics, pc_metrics):
    change = (r - p) / (r + 1e-10) * 100
    tag = f"  ↓ {change:+.1f}% improved" if change > 0 else f"  ↑ {abs(change):.1f}% worse"
    print(f"    {lbl.replace(chr(10),' '):<26}: {r:.5f}  →  {p:.5f}  {tag}")

np.save(f"{OUT_DIR}/pc_cube.npy", cube_work)

# ─────────────────────────────────────────────────────────────
# CHART 2 – BEFORE vs AFTER PHYSICAL CORRECTIONS
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
bar_chart(axes[0], raw_metrics, NOISE_LABELS, '#e05c5c', 'Raw Data (Before Corrections)')
bar_chart(axes[1], pc_metrics, NOISE_LABELS, '#5caee0', 'After Physical Corrections')
plt.suptitle('Noise Profile: Before vs After Physical Corrections', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/chart2_before_after_physical.png", dpi=150, bbox_inches='tight')
plt.close()
print("\n  [saved] chart2_before_after_physical.png")

# ─────────────────────────────────────────────────────────────
# GROUPED CHART (all in one figure for quick glance)
# ─────────────────────────────────────────────────────────────
y = np.arange(len(NOISE_LABELS))
h = 0.35
fig, ax = plt.subplots(figsize=(11, 6))
ax.barh(y - h / 2, raw_metrics, h, label='Raw Data', color='#e05c5c')
ax.barh(y + h / 2, pc_metrics, h, label='After Physical Corrections', color='#5caee0')
ax.set_yticks(y)
ax.set_yticklabels(NOISE_LABELS, fontsize=10)
ax.set_xlabel('Noise Intensity (lower = better)')
ax.set_title('Noise Profile Comparison: Raw vs Physical Corrections', fontsize=12, fontweight='bold')
ax.legend()
ax.xaxis.grid(True, linestyle='--', alpha=0.6)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/chart2_grouped.png", dpi=150)
plt.close()
print("  [saved] chart2_grouped.png")

# ─────────────────────────────────────────────────────────────
# STEP 4: REPORT
# ─────────────────────────────────────────────────────────────
report = []
report.append("# CRISM Hyperspectral Preprocessing Report\n")
report.append("## 1  Dataset Structure\n")
report.append(f"| Property | Value |\n|---|---|\n")
report.append(f"| Spatial dimensions | {LINES} Lines × {SAMPLES} Samples |\n")
report.append(f"| Spectral bands | {BANDS} |\n")
report.append(f"| Wavelength range | {WAVELENGTHS[0]:.3f} – {WAVELENGTHS[-1]:.3f} µm |\n")
report.append(f"| Raw data range | {cube_raw.min():.4f} – {cube_raw.max():.4f} W/(m² µm sr) |\n")
report.append(f"| Mean / Std | {cube_raw.mean():.4f} / {cube_raw.std():.4f} |\n\n")

report.append("## 2  Identified Noise Types\n")
table_rows = [
    ('Striping', 'High std of column means', 'Pushbroom detector column gain mismatch'),
    ('Random Noise', 'High per-band spatial std', 'Thermal read-out / photon shot noise'),
    ('Spike Noise', 'Pixels > 3σ from mean', 'Cosmic ray hits or saturated pixels'),
    ('Atmospheric Absorption', 'High spectral roughness (Δspec std)', 'CO₂/H₂O absorption in Martian atmosphere'),
    ('Dust Scattering', 'Positive slope in short-wave bands', 'Martian dust aerosol Rayleigh/Mie scattering'),
    ('Low SNR Bands', 'Bands with SNR < 1', 'Detector sensitivity limits at band edges'),
]
report.append("| Noise Type | Observable Signature | Physical Cause |\n|---|---|---|\n")
for row in table_rows:
    report.append(f"| {row[0]} | {row[1]} | {row[2]} |\n")
report.append("\n")

report.append("## 3  Physical Corrections Applied\n")
corrections = [
    ("Median Spatial Filter (3×3)", "Spike Noise, Random Noise",
     "Spike noise is fundamentally spatially isolated; a spatial median is the canonical physical method to replace anomalous pixels with their neighbourhood estimate without introducing spectral distortion.",
     "Spike Noise", raw_metrics[2], pc_metrics[2]),
    ("Per-band Radiometric Normalisation (min-max)", "Random Noise, Low SNR Bands",
     "Calibration gain/offset varies per band due to detector non-uniformity. Normalising each band independently removes this systematic offset and raises effective SNR.",
     "Random Noise", raw_metrics[1], pc_metrics[1]),
    ("Column Flat-Field Destriping", "Striping, Dust Scattering",
     "Striping is caused by column-to-column gain differences in the pushbroom detector. Dividing each column by its spatial mean across bands removes the multiplicative vertical pattern.",
     "Striping", raw_metrics[0], pc_metrics[0]),
    ("Uniform Spectral Smoothing (window=5)", "Atmospheric Absorption, Dust Scattering",
     "Atmospheric absorption bands create sharp spectral discontinuities. A short uniform filter along the wavelength axis suppresses high-frequency spectral roughness while preserving broad mineral absorption features.",
     "Atmospheric Absorption", raw_metrics[3], pc_metrics[3]),
]
for i, (name, targets, reason, primary, before, after) in enumerate(corrections, 1):
    change = (before - after) / (before + 1e-10) * 100
    report.append(f"### 3.{i}  {name}\n")
    report.append(f"- **Targets**: {targets}\n")
    report.append(f"- **Reasoning**: {reason}\n")
    report.append(f"- **Metric change ({primary})**: `{before:.5f}` → `{after:.5f}`  ({change:+.1f}%)\n\n")

report.append("## 4  Summary Table\n")
report.append("| Noise Type | Before | After | Improvement |\n|---|---|---|---|\n")
for lbl, r, p in zip(NOISE_LABELS, raw_metrics, pc_metrics):
    chg = (r - p) / (r + 1e-10) * 100
    direction = f"**↓ {chg:.1f}%**" if chg > 0 else f"↑ {abs(chg):.1f}% (amplified)"
    report.append(f"| {lbl.replace(chr(10), ' ')} | {r:.5f} | {p:.5f} | {direction} |\n")

report_path = f"{OUT_DIR}/Preprocessing_Report.md"
with open(report_path, "w") as f:
    f.writelines(report)

print(f"\n  [saved] Preprocessing_Report.md")
print("\nAll outputs saved to:", OUT_DIR)
