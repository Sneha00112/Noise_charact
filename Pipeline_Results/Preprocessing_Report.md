# CRISM Hyperspectral Preprocessing Report
## 1  Dataset Structure
| Property | Value |
|---|---|
| Spatial dimensions | 15 Lines × 64 Samples |
| Spectral bands | 107 |
| Wavelength range | 1.002 – 3.920 µm |
| Raw data range | -60.4491 – 65535.0000 W/(m² µm sr) |
| Mean / Std | 5146.1606 / 17579.8730 |

## 2  Identified Noise Types
| Noise Type | Observable Signature | Physical Cause |
|---|---|---|
| Striping | High std of column means | Pushbroom detector column gain mismatch |
| Random Noise | High per-band spatial std | Thermal read-out / photon shot noise |
| Spike Noise | Pixels > 3σ from mean | Cosmic ray hits or saturated pixels |
| Atmospheric Absorption | High spectral roughness (Δspec std) | CO₂/H₂O absorption in Martian atmosphere |
| Dust Scattering | Positive slope in short-wave bands | Martian dust aerosol Rayleigh/Mie scattering |
| Low SNR Bands | Bands with SNR < 1 | Detector sensitivity limits at band edges |

## 3  Physical Corrections Applied
### 3.1  Median Spatial Filter (3×3)
- **Targets**: Spike Noise, Random Noise
- **Reasoning**: Spike noise is fundamentally spatially isolated; a spatial median is the canonical physical method to replace anomalous pixels with their neighbourhood estimate without introducing spectral distortion.
- **Metric change (Spike Noise)**: `0.07812` → `0.01178`  (+84.9%)

### 3.2  Per-band Radiometric Normalisation (min-max)
- **Targets**: Random Noise, Low SNR Bands
- **Reasoning**: Calibration gain/offset varies per band due to detector non-uniformity. Normalising each band independently removes this systematic offset and raises effective SNR.
- **Metric change (Random Noise)**: `0.26800` → `0.11367`  (+57.6%)

### 3.3  Column Flat-Field Destriping
- **Targets**: Striping, Dust Scattering
- **Reasoning**: Striping is caused by column-to-column gain differences in the pushbroom detector. Dividing each column by its spatial mean across bands removes the multiplicative vertical pattern.
- **Metric change (Striping)**: `0.26800` → `0.00000`  (+100.0%)

### 3.4  Uniform Spectral Smoothing (window=5)
- **Targets**: Atmospheric Absorption, Dust Scattering
- **Reasoning**: Atmospheric absorption bands create sharp spectral discontinuities. A short uniform filter along the wavelength axis suppresses high-frequency spectral roughness while preserving broad mineral absorption features.
- **Metric change (Atmospheric Absorption)**: `0.00002` → `0.00000`  (+98.7%)

## 4  Summary Table
| Noise Type | Before | After | Improvement |
|---|---|---|---|
| Striping | 0.26800 | 0.00000 | **↓ 100.0%** |
| Random Noise | 0.26800 | 0.11367 | **↓ 57.6%** |
| Spike Noise | 0.07812 | 0.01178 | **↓ 84.9%** |
| Atmospheric Absorption | 0.00002 | 0.00000 | **↓ 98.7%** |
| Dust Scattering | 0.00003 | 0.00000 | **↓ 99.9%** |
| Low SNR Bands | 0.77150 | 0.19313 | **↓ 75.0%** |
