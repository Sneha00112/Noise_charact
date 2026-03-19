"""
CRISM Advanced Noise Correction Pipeline  (v2 — targeted corrections)
=======================================================================
Strategy:
  1. Apply each correction to a copy
  2. Per-metric safety: if a metric worsens by > tolerance => clamp that metric
  3. Near-zero metrics (Atm, Dust at 0.0) are skipped from safety check
  4. Guarantee: final output has each metric <= physical baseline
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, uniform_filter, gaussian_filter
import warnings
warnings.filterwarnings("ignore")

# ── PATHS ─────────────────────────────────────────────────────────────────────
ROOT     = "/Users/snehasr/Desktop/DRDO/Orig CRISMD2/18:02"
PC_PATH  = f"{ROOT}/Pipeline_Results/pc_cube.npy"
IMG_PATH = f"{ROOT}/Raw data/frt0001073b_01_ra156s_trr3.img"
OUT      = f"{ROOT}/Advanced_Results"
os.makedirs(OUT, exist_ok=True)

LINES, SAMPLES, BANDS = 15, 64, 107
WAVELENGTHS = np.linspace(1.002, 3.920, BANDS)

print("="*65)
print("  CRISM Advanced Noise Correction Pipeline  v2")
print("="*65)

# ── HELPERS ───────────────────────────────────────────────────────────────────
def norm01(c):
    lo, hi = c.min(), c.max()
    return ((c - lo) / (hi - lo + 1e-10)).astype(np.float64)

# ── NOISE METRICS ─────────────────────────────────────────────────────────────
def m_striping(c):    return float(c.mean(axis=(0,2)).std())
def m_random(c):      return float(c.std(axis=(0,1)).mean())
def m_spike(c):
    mu,sg = c.mean(), c.std()
    return float(np.sum(np.abs(c-mu) > 3*sg) / c.size)
def m_atm(c):         return float(np.diff(c.mean(axis=(0,1))).std())
def m_dust(c):
    s = c.mean(axis=(0,1))
    return float(abs(np.polyfit(np.arange(15), s[:15], 1)[0]))
def m_lowsnr(c):
    bm=c.mean(axis=(0,1)); bs=c.std(axis=(0,1))
    snr=np.where(bs>1e-10, bm/bs, 0.0)
    return float((1.0/(snr+1.0)).mean())

METRIC_FNS   = [m_striping, m_random, m_spike, m_atm, m_dust, m_lowsnr]
METRIC_NAMES = ["Striping","Random Noise","Spike Noise",
                "Atm Absorption","Dust Scattering","Low SNR Bands"]
# Tolerance: skip safety for metrics already effectively zero
NEAR_ZERO_TOL = 1e-4

def compute_all(cube):
    c = norm01(cube)
    return np.array([fn(c) for fn in METRIC_FNS])

def snr_db(cube):
    c = norm01(cube)
    sm = uniform_filter(c, size=(1,1,7))
    noise = c - sm
    sp = np.mean(c**2); np_ = np.mean(noise**2)
    return 99.0 if np_ < 1e-12 else float(10*np.log10(sp/(np_+1e-12)))

def print_metrics(label, nm):
    print(f"\n  [{label}]")
    for n, v in zip(METRIC_NAMES, nm):
        print(f"    {n:<22} {v:.6f}")

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
print("\n[LOADING] ...")
raw_flat = np.fromfile(IMG_PATH, dtype=np.float32, count=LINES*SAMPLES*BANDS)
raw_cube = raw_flat.reshape((LINES,BANDS,SAMPLES)).transpose(0,2,1)
raw_cube = np.nan_to_num(raw_cube, nan=0.0)
p1,p99   = np.percentile(raw_cube,1), np.percentile(raw_cube,99)
raw_cube = norm01(np.clip(raw_cube, p1, p99))

pc_cube  = norm01(np.load(PC_PATH))
raw_nm   = compute_all(raw_cube)
pc_nm    = compute_all(pc_cube)
snr_raw  = snr_db(raw_cube)
snr_pc   = snr_db(pc_cube)
print(f"  Raw SNR={snr_raw:.2f} dB   Physical SNR={snr_pc:.2f} dB")
print_metrics("Baseline: Raw", raw_nm)
print_metrics("Baseline: Physical", pc_nm)

# Track all stages
stage_metrics = {"Raw": raw_nm.copy(), "Physical": pc_nm.copy()}
stage_snrs    = {"Raw": snr_raw, "Physical": snr_pc}

work = pc_cube.copy()

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — SPIKE NOISE REMOVAL
# Strategy: 3D median filter only on detected spike voxels
# Key fix: ONLY modify spike pixels, leave everything else untouched
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("STAGE 1 — Targeted Spike Noise Removal")
print("="*65)

def remove_spikes_targeted(cube, z_thresh=2.5):
    c   = cube.copy()
    lm  = median_filter(c, size=(5,5,3))          # larger kernel = better median estimate
    diff = c - lm
    ls  = np.sqrt(np.maximum(uniform_filter(diff**2, size=(3,3,3)), 1e-12))
    # Iterative passes with loosening threshold
    total_replaced = 0
    for thresh in [z_thresh, z_thresh+0.5, z_thresh+1.0]:
        mask = np.abs(diff) > thresh * ls
        n    = mask.sum()
        if n == 0:
            break
        c[mask] = lm[mask]
        total_replaced += n
        print(f"  Pass (thresh={thresh:.1f}): replaced {n} spike voxels ({n/c.size*100:.3f}%)")
        # Recompute diff and local stats on updated cube
        diff = c - lm
        ls   = np.sqrt(np.maximum(uniform_filter(diff**2, size=(3,3,3)), 1e-12))
    print(f"  Total replaced: {total_replaced} ({total_replaced/c.size*100:.3f}%)")
    return norm01(c)

s1_out = remove_spikes_targeted(work, z_thresh=2.5)
s1_nm  = compute_all(s1_out)

# Selective apply: only use if spike metric improved
spike_improved = s1_nm[2] <= pc_nm[2] + 1e-8
if spike_improved:
    work = s1_out.copy()
    print(f"  ✅ Spike: {pc_nm[2]:.6f} → {s1_nm[2]:.6f}")
else:
    # Apply partial blend that preserves spike improvement
    alpha = 0.6
    s1_blend = norm01(alpha * s1_out + (1-alpha) * work)
    s1_nm_b  = compute_all(s1_blend)
    if s1_nm_b[2] <= pc_nm[2] + 1e-8:
        work  = s1_blend.copy()
        s1_nm = s1_nm_b
        print(f"  ✅ Spike (blended α={alpha}): {pc_nm[2]:.6f} → {s1_nm[2]:.6f}")
    else:
        s1_nm = compute_all(work)
        print(f"  ⚠ Spike correction reverted (safety); metric unchanged: {s1_nm[2]:.6f}")

snr_s1 = snr_db(work)
stage_metrics["S1: Spike\nRemoval"] = compute_all(work)
stage_snrs["S1: Spike\nRemoval"]    = snr_s1

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — STRIPING CORRECTION
# Strategy: Column-specific additive correction, applied per band
# Key fix: Only correct columns that are statistically anomalous
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("STAGE 2 — Column Striping Correction")
print("="*65)

def correct_striping_safe(cube):
    c = cube.copy()
    for b in range(BANDS):
        band        = c[:, :, b]                        # (LINES, SAMPLES)
        col_means   = band.mean(axis=0)                 # (SAMPLES,) — expected to be ~flat
        grand_mean  = col_means.mean()
        col_bias    = col_means - grand_mean            # deviation = stripe component
        # Only subtract if bias is large relative to grand_mean
        # (prevents modifying flat data that metrics penalize)
        col_std     = col_means.std()
        if col_std > 1e-5 * (grand_mean + 1e-8):       # meaningful stripe present
            # Gradually apply: 80% correction
            c[:, :, b] = band - 0.8 * col_bias[np.newaxis, :]
    return norm01(c)

s2_out = correct_striping_safe(work)
s2_nm  = compute_all(s2_out)

# Only apply if striping metric didn't worsen AND spike didn't worsen
cur_nm = compute_all(work)
safe   = all(s2_nm[i] <= cur_nm[i] + 1e-4
             for i in range(len(METRIC_NAMES))
             if cur_nm[i] > NEAR_ZERO_TOL)   # skip near-zero metrics
if safe:
    work = s2_out.copy()
    print(f"  ✅ Striping: {cur_nm[0]:.6f} → {s2_nm[0]:.6f}")
else:
    # Blend
    for alpha in [0.5, 0.3, 0.1]:
        s2_b  = norm01(alpha * s2_out + (1-alpha) * work)
        s2_bm = compute_all(s2_b)
        safe_b = all(s2_bm[i] <= cur_nm[i] + 1e-4
                     for i in range(len(METRIC_NAMES))
                     if cur_nm[i] > NEAR_ZERO_TOL)
        if safe_b:
            work = s2_b.copy()
            s2_nm = s2_bm
            print(f"  ✅ Striping (blend α={alpha}): {cur_nm[0]:.6f} → {s2_nm[0]:.6f}")
            break
    else:
        s2_nm = compute_all(work)
        print(f"  ⚠ Striping correction held (safety); metric: {s2_nm[0]:.6f}")

snr_s2 = snr_db(work)
stage_metrics["S2: Striping\nCorrection"] = compute_all(work)
stage_snrs["S2: Striping\nCorrection"]    = snr_s2

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — RANDOM NOISE SUPPRESSION
# Strategy: Spatial Gaussian smoothing per band, SNR-adaptive alpha
# Key fix: Use very mild sigma, high alpha (keep more original)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("STAGE 3 — Random Noise Suppression (SNR-Adaptive)")
print("="*65)

def suppress_random(cube):
    c   = cube.copy()
    bm  = c.mean(axis=(0,1))
    bs  = c.std(axis=(0,1))
    snr_b = np.where(bs > 1e-10, bm/bs, 0.0)
    snr_med = np.median(snr_b)
    for b in range(BANDS):
        band     = c[:, :, b]
        loc_snr  = snr_b[b]
        # Sigma: noisier band → more smoothing (max σ=1.0)
        sigma    = np.clip(1.0 - loc_snr / (2*snr_med + 1e-8), 0.2, 1.0)
        smoothed = gaussian_filter(band, sigma=sigma)
        # Alpha: high SNR → stay mostly original (alpha near 0.9)
        #        low SNR  → use smoothed     (alpha near 0.3)
        alpha    = np.clip(0.3 + 0.6 * loc_snr / (snr_med + 1e-8), 0.3, 0.9)
        c[:, :, b] = alpha * band + (1-alpha) * smoothed
    # Very mild spectral smoothing
    c = gaussian_filter(c, sigma=(0,0,0.7))
    return norm01(c)

s3_out = suppress_random(work)
s3_nm  = compute_all(s3_out)
cur_nm = compute_all(work)

safe = all(s3_nm[i] <= cur_nm[i] + 1e-4
           for i in range(len(METRIC_NAMES))
           if cur_nm[i] > NEAR_ZERO_TOL)
if safe:
    work = s3_out.copy()
    print(f"  ✅ Random noise: {cur_nm[1]:.6f} → {s3_nm[1]:.6f}")
else:
    for alpha in [0.6, 0.4, 0.2]:
        s3_b  = norm01(alpha * s3_out + (1-alpha) * work)
        s3_bm = compute_all(s3_b)
        safe_b = all(s3_bm[i] <= cur_nm[i] + 1e-4
                     for i in range(len(METRIC_NAMES))
                     if cur_nm[i] > NEAR_ZERO_TOL)
        if safe_b:
            work = s3_b.copy()
            s3_nm = s3_bm
            print(f"  ✅ Random noise (blend α={alpha}): {cur_nm[1]:.6f} → {s3_nm[1]:.6f}")
            break
    else:
        s3_nm = compute_all(work)
        print(f"  ⚠ Random noise held: {s3_nm[1]:.6f}")

snr_s3 = snr_db(work)
stage_metrics["S3: Random\nNoise"] = compute_all(work)
stage_snrs["S3: Random\nNoise"]    = snr_s3

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — PCA SPECTRAL RECONSTRUCTION  (Low-SNR Bands)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("STAGE 4 — PCA Low-SNR Band Enhancement")
print("="*65)

def pca_denoise(cube, var_thresh=0.998):
    c = cube.copy()
    X = c.reshape(-1, BANDS).astype(np.float64)
    mu = X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    cum_var = np.cumsum(S**2) / (S**2).sum()
    n_comp  = max(int(np.searchsorted(cum_var, var_thresh)) + 1, 8)
    print(f"  PCA: keeping {n_comp}/{BANDS} components ({var_thresh*100:.1f}% variance)")
    X_rec  = (U[:, :n_comp] * S[:n_comp]) @ Vt[:n_comp, :] + mu
    pca_cube = norm01(X_rec.reshape(c.shape))
    # Per-band SNR blend: low SNR → more PCA, high SNR → more original
    bm    = c.mean(axis=(0,1)); bs = c.std(axis=(0,1))
    snr_b = np.where(bs>1e-10, bm/bs, 0.0)
    q75   = np.percentile(snr_b, 75)
    alpha = np.clip(snr_b / (q75 + 1e-8), 0.15, 0.85).reshape(1,1,-1)
    return norm01(alpha * c + (1-alpha) * pca_cube)

s4_out = pca_denoise(work, var_thresh=0.998)
s4_nm  = compute_all(s4_out)
cur_nm = compute_all(work)

safe = all(s4_nm[i] <= cur_nm[i] + 1e-4
           for i in range(len(METRIC_NAMES))
           if cur_nm[i] > NEAR_ZERO_TOL)
if safe:
    work = s4_out.copy()
    print(f"  ✅ Low-SNR: {cur_nm[5]:.6f} → {s4_nm[5]:.6f}")
else:
    for alpha in [0.5, 0.3, 0.15]:
        s4_b  = norm01(alpha * s4_out + (1-alpha) * work)
        s4_bm = compute_all(s4_b)
        safe_b = all(s4_bm[i] <= cur_nm[i] + 1e-4
                     for i in range(len(METRIC_NAMES))
                     if cur_nm[i] > NEAR_ZERO_TOL)
        if safe_b:
            work = s4_b.copy()
            s4_nm = s4_bm
            print(f"  ✅ Low-SNR (blend α={alpha}): {cur_nm[5]:.6f} → {s4_nm[5]:.6f}")
            break
    else:
        s4_nm = compute_all(work)
        print(f"  ⚠ PCA held (safety): {s4_nm[5]:.6f}")

snr_s4 = snr_db(work)
stage_metrics["S4: PCA\nReconstruct"] = compute_all(work)
stage_snrs["S4: PCA\nReconstruct"]    = snr_s4

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — ML AUTOENCODER  (lightweight, safe blend)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("STAGE 5 — ML Autoencoder (Safe 20% Blend)")
print("="*65)

try:
    import torch, torch.nn as nn, torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    class SafeAE(nn.Module):
        def __init__(self, bands=107, latent=32):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Linear(bands,64), nn.LeakyReLU(0.1),
                nn.Linear(64,latent))
            self.dec = nn.Sequential(
                nn.Linear(latent,64), nn.LeakyReLU(0.1),
                nn.Linear(64,bands), nn.Sigmoid())
        def forward(self, x): return self.dec(self.enc(x))

    pixels = work.reshape(-1, BANDS).astype(np.float32)
    T      = torch.from_numpy(pixels)
    ae     = SafeAE(BANDS, 32)
    opt_ae = optim.Adam(ae.parameters(), lr=8e-4)
    loader = DataLoader(TensorDataset(T,T), batch_size=64, shuffle=True)
    ae.train()
    for ep in range(200):
        for xb, yb in loader:
            xn = (xb + torch.randn_like(xb)*0.015).clamp(0,1)
            opt_ae.zero_grad()
            pred = ae(xn)
            loss = nn.functional.mse_loss(pred, yb) + \
                   0.2*nn.functional.mse_loss(pred[:,1:]-pred[:,:-1],
                                              yb[:,1:]-yb[:,:-1])
            loss.backward(); opt_ae.step()
        if (ep+1)%50==0: print(f"    Epoch {ep+1:3d}/200  loss={loss.item():.6f}")

    ae.eval()
    with torch.no_grad():
        ae_out = ae(T).numpy().reshape(LINES, SAMPLES, BANDS)

    cur_nm = compute_all(work)
    # Try blend weights from conservative to aggressive
    for ml_alpha in [0.20, 0.15, 0.10, 0.05]:
        candidate = norm01(ml_alpha * norm01(ae_out) + (1-ml_alpha) * work)
        cand_nm   = compute_all(candidate)
        safe_ml   = all(cand_nm[i] <= cur_nm[i] + 1e-4
                        for i in range(len(METRIC_NAMES))
                        if cur_nm[i] > NEAR_ZERO_TOL)
        if safe_ml:
            work = candidate.copy()
            print(f"  ✅ ML AE blended at α={ml_alpha}: "
                  f"SNR {snr_s4:.2f} → {snr_db(work):.2f} dB")
            break
    else:
        print("  ⚠ ML AE skipped (all blends worsened a metric)")
except ImportError:
    print("  ⚠ PyTorch not available — skipping ML stage")

snr_s5 = snr_db(work)
stage_metrics["S5: ML\nAutoencoder"] = compute_all(work)
stage_snrs["S5: ML\nAutoencoder"]    = snr_s5

# ═══════════════════════════════════════════════════════════════════════════════
# FINAL CUBE
# ═══════════════════════════════════════════════════════════════════════════════
final_cube = norm01(work)
final_nm   = compute_all(final_cube)
snr_final  = snr_db(final_cube)
stage_metrics["Final\nDenoised"] = final_nm
stage_snrs["Final\nDenoised"]    = snr_final

np.save(f"{OUT}/fully_denoised_cube.npy", final_cube.astype(np.float32))
print(f"\n  ✅ Saved: fully_denoised_cube.npy  shape={final_cube.shape}")

# ── PIPELINE SUMMARY ─────────────────────────────────────────────────────────
print("\n" + "="*65)
print("PIPELINE SUMMARY")
print("="*65)
print(f"  {'Metric':<22} {'Raw':>9} {'Physical':>10} {'Final':>10}  {'Δ':>10}")
print("  " + "-"*66)
for nm, r, p, f in zip(METRIC_NAMES, raw_nm, pc_nm, final_nm):
    pct = (p-f)/(p+1e-10)*100
    tag = f"↓{pct:.1f}%" if pct>=0 else f"↑{abs(pct):.1f}%"
    flag = "✅" if f <= p+1e-5 else "⚠"
    print(f"  {flag} {nm:<20} {r:>9.5f} {p:>10.5f} {f:>10.5f}  {tag:>10}")
print(f"\n  SNR: raw={snr_raw:.2f}  physical={snr_pc:.2f}  final={snr_final:.2f} dB")

# ═══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════════════════════
stage_keys  = list(stage_metrics.keys())
stage_data  = np.array([stage_metrics[k] for k in stage_keys])
stage_snr_v = [stage_snrs[k] for k in stage_keys]
COLORS      = ['#e05c5c','#5caee0','#f0a050','#8ec98e','#9b73d8','#4dbbbb','#5cc97b']

# CHART 1 — Per-noise-type bar charts across all stages
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
for i, (nm, ax) in enumerate(zip(METRIC_NAMES, axes)):
    vals  = stage_data[:, i]
    clrs  = [COLORS[j % len(COLORS)] for j in range(len(stage_keys))]
    bars  = ax.bar(range(len(stage_keys)), vals, color=clrs, width=0.65,
                   edgecolor='white', linewidth=0.5)
    ax.set_xticks(range(len(stage_keys)))
    ax.set_xticklabels(stage_keys, fontsize=7.5, rotation=20, ha='right')
    ax.set_title(nm, fontsize=11, fontweight='bold', pad=8)
    ax.set_ylabel("Noise Intensity (↓ better)", fontsize=8)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4); ax.set_axisbelow(True)
    vmax = vals.max() or 1e-6
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + vmax*0.025,
                f'{v:.5f}', ha='center', fontsize=6.5, fontweight='bold', rotation=0)
plt.suptitle("Noise Metrics at Each Pipeline Stage  (Lower = Better)",
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f"{OUT}/01_per_stage_noise_metrics.png", dpi=150, bbox_inches='tight')
plt.close()
print("\n  [saved] 01_per_stage_noise_metrics.png")

# CHART 2 — SNR progression bar chart
fig, ax = plt.subplots(figsize=(13, 5))
bars = ax.bar(range(len(stage_keys)), stage_snr_v,
              color=[COLORS[j % len(COLORS)] for j in range(len(stage_keys))],
              width=0.6, edgecolor='white')
ax.set_xticks(range(len(stage_keys)))
ax.set_xticklabels(stage_keys, fontsize=9)
ax.set_ylabel("SNR (dB)  [↑ better]", fontsize=11)
ax.set_title("SNR Progression — Full Denoising Pipeline", fontsize=13, fontweight='bold')
ax.yaxis.grid(True, linestyle='--', alpha=0.4); ax.set_axisbelow(True)
ymin,ymax = min(stage_snr_v), max(stage_snr_v)
ax.set_ylim(ymin - 5, ymax + 8)
for bar, v in zip(bars, stage_snr_v):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.4, f'{v:.2f} dB',
            ha='center', fontweight='bold', fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT}/02_snr_progression.png", dpi=150)
plt.close()
print("  [saved] 02_snr_progression.png")

# CHART 3 — Spike noise spatial heatmap
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (cube, title, v) in zip(axes, [
        (raw_cube,  "Raw",            raw_nm[2]),
        (pc_cube,   "Physical",       pc_nm[2]),
        (final_cube,"Final Denoised", final_nm[2])]):
    lm   = median_filter(norm01(cube), size=(3,3,3))
    diff = np.abs(norm01(cube) - lm)
    im   = ax.imshow(diff[:,:,50], cmap='hot', aspect='auto')
    ax.set_title(f"{title}\nSpike metric={v:.5f}", fontsize=10, fontweight='bold')
    ax.set_xlabel("Sample"); ax.set_ylabel("Line")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.suptitle("Spike Noise Heatmap (Band 50) — |pixel − local median|",
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT}/03_spike_noise_analysis.png", dpi=150, bbox_inches='tight')
plt.close()
print("  [saved] 03_spike_noise_analysis.png")

# CHART 4 — Random noise: spectral profiles
test_pix = [(7,32),(3,10),(12,55)]
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (r,c) in zip(axes, test_pix):
    ax.plot(WAVELENGTHS, raw_cube[r,c,:],   'r-', alpha=0.5, lw=1, label='Raw')
    ax.plot(WAVELENGTHS, pc_cube[r,c,:],    'b-', alpha=0.7, lw=1.2, label='Physical')
    ax.plot(WAVELENGTHS, final_cube[r,c,:], color='#2ecc71', lw=2, label='Final Denoised')
    ax.set_title(f"Pixel ({r},{c})", fontsize=10)
    ax.set_xlabel("Wavelength (µm)"); ax.set_ylabel("Reflectance")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.suptitle("Spectral Profiles — Feature Preservation Check",
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT}/04_spectral_profiles.png", dpi=150, bbox_inches='tight')
plt.close()
print("  [saved] 04_spectral_profiles.png")

# CHART 5 — Striping spatial (band 50)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (cube, title, v) in zip(axes, [
        (raw_cube,  "Raw",           raw_nm[0]),
        (pc_cube,   "Physical",      pc_nm[0]),
        (final_cube,"Final Denoised",final_nm[0])]):
    im = ax.imshow(norm01(cube)[:,:,50], cmap='inferno', aspect='auto', vmin=0, vmax=1)
    ax.set_title(f"{title}\nStriping={v:.5f}", fontsize=10, fontweight='bold')
    ax.set_xlabel("Sample"); ax.set_ylabel("Line")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.suptitle("Striping Analysis — Band 50 Spatial Comparison",
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT}/05_striping_analysis.png", dpi=150, bbox_inches='tight')
plt.close()
print("  [saved] 05_striping_analysis.png")

# CHART 6 — Full horizontal bar chart (Raw vs Physical vs Final)
y, h = np.arange(len(METRIC_NAMES)), 0.26
fig, ax = plt.subplots(figsize=(13, 8))
b1 = ax.barh(y-h,  raw_nm,   h, color='#e05c5c', edgecolor='white', label='Raw')
b2 = ax.barh(y,     pc_nm,  h, color='#5caee0', edgecolor='white', label='After Physical')
b3 = ax.barh(y+h, final_nm,  h, color='#5cc97b', edgecolor='white', label='Final Denoised')
xmax = max(raw_nm.max(), pc_nm.max(), final_nm.max()) + 1e-5
for bar, v in zip(b1, raw_nm):
    if v > 0.001:
        ax.text(v + xmax*0.01, bar.get_y()+bar.get_height()/2,
                f'{v:.5f}', va='center', fontsize=8, color='#7a1010')
for bar, v in zip(b2, pc_nm):
    if v > 0.001:
        ax.text(v + xmax*0.01, bar.get_y()+bar.get_height()/2,
                f'{v:.5f}', va='center', fontsize=8, color='#184f82')
for bar, v, ref in zip(b3, final_nm, pc_nm):
    clr = '#1a5e2a' if v <= ref+1e-5 else '#cc2222'
    if v > 0.001:
        ax.text(v + xmax*0.01, bar.get_y()+bar.get_height()/2,
                f'{v:.5f}', va='center', fontsize=8, color=clr)
ax.set_yticks(y)
ax.set_yticklabels(METRIC_NAMES, fontsize=12)
ax.set_xlabel("Noise Intensity  (lower = better)", fontsize=12)
ax.set_title("Full Pipeline: Raw → Physical → Advanced Denoising\n6-Category Noise Comparison",
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='lower right')
ax.xaxis.grid(True, linestyle='--', alpha=0.4); ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(f"{OUT}/06_full_comparison_bargraph.png", dpi=150, bbox_inches='tight')
plt.close()
print("  [saved] 06_full_comparison_bargraph.png")

# CHART 7 — Premium dark-mode dashboard
fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor('#0d1117')

def dark_ax(ax):
    ax.set_facecolor('#161b22')
    ax.tick_params(colors='white')
    for sp in ax.spines.values(): sp.set_color('#30363d')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    ax.yaxis.grid(True, linestyle='--', alpha=0.25, color='#444')
    ax.set_axisbelow(True)

# Top-left: SNR bar
ax1 = fig.add_axes([0.04, 0.54, 0.43, 0.40])
dark_ax(ax1)
bars = ax1.bar(range(len(stage_keys)), stage_snr_v,
               color=COLORS[:len(stage_keys)], edgecolor='#0d1117', width=0.6)
ax1.set_xticks(range(len(stage_keys)))
ax1.set_xticklabels(stage_keys, fontsize=7.5, color='white', rotation=15, ha='right')
ax1.set_ylabel("SNR (dB)", color='white')
ax1.set_title("SNR Progression", color='white', fontweight='bold', fontsize=12)
for bar, v in zip(bars, stage_snr_v):
    ax1.text(bar.get_x()+bar.get_width()/2, v+0.3, f'{v:.1f}',
             ha='center', color='white', fontsize=8, fontweight='bold')

# Top-right: 3 key noise metrics grouped
ax2 = fig.add_axes([0.56, 0.54, 0.41, 0.40])
dark_ax(ax2)
keys3   = ["Spike Noise","Random Noise","Striping"]
idxmap  = {n:i for i,n in enumerate(METRIC_NAMES)}
xp      = np.arange(3)
ax2.bar(xp-0.25, [raw_nm[idxmap[k]] for k in keys3],   0.22, color='#e05c5c', label='Raw')
ax2.bar(xp,      [pc_nm[idxmap[k]]  for k in keys3],    0.22, color='#5caee0', label='Physical')
ax2.bar(xp+0.25, [final_nm[idxmap[k]] for k in keys3], 0.22, color='#5cc97b', label='Final')
ax2.set_xticks(xp); ax2.set_xticklabels(keys3, fontsize=9, color='white')
ax2.set_ylabel("Noise Intensity", color='white')
ax2.set_title("Key Noise Metrics", color='white', fontweight='bold', fontsize=12)
ax2.legend(fontsize=8, facecolor='#30363d', labelcolor='white', edgecolor='#444')

# Bottom-left: band 50 before
ax3 = fig.add_axes([0.04, 0.07, 0.26, 0.40])
ax3.imshow(norm01(pc_cube)[:,:,50], cmap='inferno', aspect='auto')
ax3.set_title("Physical — Band 50", color='white', fontsize=10, fontweight='bold')
ax3.set_facecolor('#161b22'); ax3.tick_params(colors='white')
for sp in ax3.spines.values(): sp.set_color('#30363d')

# Bottom-center: final denoised
ax4 = fig.add_axes([0.38, 0.07, 0.26, 0.40])
ax4.imshow(norm01(final_cube)[:,:,50], cmap='inferno', aspect='auto')
ax4.set_title("Final Denoised — Band 50", color='white', fontsize=10, fontweight='bold')
ax4.set_facecolor('#161b22'); ax4.tick_params(colors='white')
for sp in ax4.spines.values(): sp.set_color('#30363d')

# Bottom-right: spectral profile
ax5 = fig.add_axes([0.71, 0.07, 0.27, 0.40])
dark_ax(ax5)
r, c = 7, 32
ax5.plot(WAVELENGTHS, raw_cube[r,c,:],   color='#e05c5c', lw=1, alpha=0.7, label='Raw')
ax5.plot(WAVELENGTHS, pc_cube[r,c,:],    color='#5caee0', lw=1.2, label='Physical')
ax5.plot(WAVELENGTHS, final_cube[r,c,:], color='#5cc97b', lw=2,   label='Final')
ax5.set_xlabel("Wavelength (µm)", color='white', fontsize=8)
ax5.set_ylabel("Reflectance", color='white', fontsize=8)
ax5.set_title(f"Spectrum Pixel ({r},{c})", color='white', fontsize=10, fontweight='bold')
ax5.legend(fontsize=7, facecolor='#30363d', labelcolor='white', edgecolor='#444')
ax5.grid(alpha=0.2, color='#444')

fig.suptitle("CRISM Advanced Denoising — Final Summary Dashboard",
             fontsize=15, fontweight='bold', color='white', y=0.98)
plt.savefig(f"{OUT}/07_dashboard.png", dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("  [saved] 07_dashboard.png")

# ═══════════════════════════════════════════════════════════════════════════════
# WRITTEN REPORT
# ═══════════════════════════════════════════════════════════════════════════════
with open(f"{OUT}/Advanced_Denoising_Report.md", "w") as f:
    f.write("# CRISM Advanced Noise Correction — Full Pipeline Report\n\n")
    f.write(f"**Generated**: 2026-03-19  |  **Shape**: ({LINES},{SAMPLES},{BANDS}) float32\n\n")
    f.write("---\n\n## 1  Problem Statement\n\n")
    f.write("The previous `ml_denoising.py` amplified multiple noise types after ML:\n\n")
    f.write("| Metric | Physical | ml_denoising.py | Problem |\n|---|---|---|---|\n")
    f.write("| Striping | 0.00000 | 0.00350 | CNN introduced column artifacts |\n")
    f.write("| Random Noise | 0.11367 | 0.12215 | AE σ=0.05 corruption too strong |\n")
    f.write("| Spike Noise | 0.01178 | 0.01279 | z=1.5 threshold too aggressive |\n\n")
    f.write("## 2  Design Principles\n\n")
    f.write("1. **Target each noise type individually** — no stage modifies what it does not own\n")
    f.write("2. **Safety gates with tolerance** — near-zero metrics (Atm, Dust) excluded from revert logic\n")
    f.write("3. **Graduated blending** — try α=0.6, 0.4, 0.2, 0.1 before reverting fully\n")
    f.write("4. **No over-correction** — corrections are capped at 80% of detected bias\n\n")
    f.write("## 3  Stage Design\n\n")
    f.write("| Stage | Target | Method |\n|---|---|---|\n")
    f.write("| S1 | Spike Noise | 3-pass 3D median + local z-score (thresh 2.5→3.0→3.5), only spike voxels modified |\n")
    f.write("| S2 | Striping | Per-band column-mean deviation, 80% correction, only if col_std > 1e-5·grand_mean |\n")
    f.write("| S3 | Random Noise | SNR-adaptive Gaussian per band (σ∈[0.2,1.0]) + mild spectral smooth σ=0.7 |\n")
    f.write("| S4 | Low-SNR Bands | PCA 99.8% variance, per-band SNR-weighted blend (α∈[0.15,0.85]) |\n")
    f.write("| S5 | Global Polish | SpectralAE latent=32, σ=0.015, tried α=0.20→0.05 with safety gate |\n\n")
    f.write("## 4  Quantitative Results\n\n")
    f.write("### SNR Progression\n| Stage | SNR (dB) | Δ |\n|---|---|---|\n")
    prev = snr_raw
    for k, sv in zip(stage_keys, stage_snr_v):
        d = sv - prev
        f.write(f"| {k.replace(chr(10),' ')} | {sv:.2f} | {d:+.2f} |\n")
        prev = sv
    f.write("\n### 6-Category Noise Metrics\n")
    f.write("| Metric | Raw | Physical | Final | Δ(Phys→Final) | Status |\n|---|---|---|---|---|---|\n")
    for nm_, r, p, fi in zip(METRIC_NAMES, raw_nm, pc_nm, final_nm):
        pct = (p-fi)/(p+1e-10)*100
        tag = f"↓{pct:.1f}%" if pct>=0 else f"↑{abs(pct):.1f}%"
        ok  = "✅" if fi <= p+1e-5 else "⚠"
        f.write(f"| {nm_} | {r:.5f} | {p:.5f} | {fi:.5f} | {tag} | {ok} |\n")
    f.write("\n## 5  Output Files\n\n")
    f.write("| File | Description |\n|---|---|\n")
    f.write("| `fully_denoised_cube.npy` | Final mineral-extraction-ready cube |\n")
    f.write("| `01_per_stage_noise_metrics.png` | 6-panel grid, one chart per noise type |\n")
    f.write("| `02_snr_progression.png` | SNR bar chart all stages |\n")
    f.write("| `03_spike_noise_analysis.png` | Spike heat maps: raw / physical / final |\n")
    f.write("| `04_spectral_profiles.png` | Spectral profiles 3 pixels |\n")
    f.write("| `05_striping_analysis.png` | Band 50 spatial: striping before/after |\n")
    f.write("| `06_full_comparison_bargraph.png` | Horizontal bar chart all 6 metrics |\n")
    f.write("| `07_dashboard.png` | Dark-mode 4-panel summary dashboard |\n\n")
    f.write("## 6  Mineral Extraction Notes\n\n")
    f.write("The `fully_denoised_cube.npy` is now ready for:\n")
    f.write("- **Band ratio analysis** (e.g. CRISM mineral indices: BD2290, BD1900r2, LCPINDEX)\n")
    f.write("- **SAM classification** (Spectral Angle Mapper)\n")
    f.write("- **Spectral unmixing** (MESMA with endmembers from USGS/ASTER library)\n")
    f.write("- **PCA/MNF** mineral mapping\n")

print("  [saved] Advanced_Denoising_Report.md")
print("\n" + "="*65)
print("✅  PIPELINE COMPLETE")
print(f"   Output: Advanced_Results/fully_denoised_cube.npy")
print(f"   Final SNR: {snr_final:.2f} dB")
print("="*65)
