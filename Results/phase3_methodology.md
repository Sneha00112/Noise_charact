# Phase 3 Methodology: Hybrid Preprocessing

## 3.1 Physical Corrections
- **Radiometric Normalization:** Data scaled to [0,1].
- **Illumination Normalization:** Corrected cross-track lighting gradients.
- **Atmospheric Correction:** Log Residuals method applied to remove global atmospheric transmission effects.

## 3.2 ML-Based Denoising
- **Autoencoder:** Used a 4-layer dense Autoencoder (64->32->64) to remove high-frequency random thermal noise across the spectral domain.
- **CNN:** A 1D Convolutional Neural Network with Total Variation (TV) loss applied along the spatial dimension to suppress vertical striping artifacts common in pushbroom sensors.

## Validation Tracking
- **SNR Improvement:** SNR per band generally increased and became smoother, particularly in low-signal VNIR/SWIR regions.
- **Absorption Preservation:** As shown in `before_after_spectra.png`, large-scale mineral absorption bands are preserved without the small chaotic noise fluctuations.
