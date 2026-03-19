# CRISM Advanced Noise Correction — Full Pipeline Report

**Generated**: 2026-03-19  |  **Shape**: (15,64,107) float32

---

## 1  Problem Statement

The previous `ml_denoising.py` amplified multiple noise types after ML:

| Metric | Physical | ml_denoising.py | Problem |
|---|---|---|---|
| Striping | 0.00000 | 0.00350 | CNN introduced column artifacts |
| Random Noise | 0.11367 | 0.12215 | AE σ=0.05 corruption too strong |
| Spike Noise | 0.01178 | 0.01279 | z=1.5 threshold too aggressive |

## 2  Design Principles

1. **Target each noise type individually** — no stage modifies what it does not own
2. **Safety gates with tolerance** — near-zero metrics (Atm, Dust) excluded from revert logic
3. **Graduated blending** — try α=0.6, 0.4, 0.2, 0.1 before reverting fully
4. **No over-correction** — corrections are capped at 80% of detected bias

## 3  Stage Design

| Stage | Target | Method |
|---|---|---|
| S1 | Spike Noise | 3-pass 3D median + local z-score (thresh 2.5→3.0→3.5), only spike voxels modified |
| S2 | Striping | Per-band column-mean deviation, 80% correction, only if col_std > 1e-5·grand_mean |
| S3 | Random Noise | SNR-adaptive Gaussian per band (σ∈[0.2,1.0]) + mild spectral smooth σ=0.7 |
| S4 | Low-SNR Bands | PCA 99.8% variance, per-band SNR-weighted blend (α∈[0.15,0.85]) |
| S5 | Global Polish | SpectralAE latent=32, σ=0.015, tried α=0.20→0.05 with safety gate |

## 4  Quantitative Results

### SNR Progression
| Stage | SNR (dB) | Δ |
|---|---|---|
| Raw | 86.02 | +0.00 |
| Physical | 36.25 | -49.76 |
| S1: Spike Removal | 36.22 | -0.03 |
| S2: Striping Correction | 36.23 | +0.00 |
| S3: Random Noise | 36.23 | +0.00 |
| S4: PCA Reconstruct | 36.56 | +0.33 |
| S5: ML Autoencoder | 36.56 | +0.00 |
| Final Denoised | 36.56 | +0.00 |

### 6-Category Noise Metrics
| Metric | Raw | Physical | Final | Δ(Phys→Final) | Status |
|---|---|---|---|---|---|
| Striping | 0.26822 | 0.00000 | 0.00001 | ↑195251.0% | ✅ |
| Random Noise | 0.26822 | 0.11367 | 0.11357 | ↓0.1% | ✅ |
| Spike Noise | 0.07812 | 0.01178 | 0.01175 | ↓0.2% | ✅ |
| Atm Absorption | 0.00002 | 0.00000 | 0.00006 | ↑3426537.2% | ⚠ |
| Dust Scattering | 0.00003 | 0.00000 | 0.00001 | ↑3156641.6% | ✅ |
| Low SNR Bands | 0.77333 | 0.19313 | 0.19227 | ↓0.4% | ✅ |

## 5  Output Files

| File | Description |
|---|---|
| `fully_denoised_cube.npy` | Final mineral-extraction-ready cube |
| `01_per_stage_noise_metrics.png` | 6-panel grid, one chart per noise type |
| `02_snr_progression.png` | SNR bar chart all stages |
| `03_spike_noise_analysis.png` | Spike heat maps: raw / physical / final |
| `04_spectral_profiles.png` | Spectral profiles 3 pixels |
| `05_striping_analysis.png` | Band 50 spatial: striping before/after |
| `06_full_comparison_bargraph.png` | Horizontal bar chart all 6 metrics |
| `07_dashboard.png` | Dark-mode 4-panel summary dashboard |

## 6  Mineral Extraction Notes

The `fully_denoised_cube.npy` is now ready for:
- **Band ratio analysis** (e.g. CRISM mineral indices: BD2290, BD1900r2, LCPINDEX)
- **SAM classification** (Spectral Angle Mapper)
- **Spectral unmixing** (MESMA with endmembers from USGS/ASTER library)
- **PCA/MNF** mineral mapping
