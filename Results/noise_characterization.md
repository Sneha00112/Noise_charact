# Mission-wise Noise Characterization

## Categorization of Noise
- **Random Noise**: Evident in low SNR bands, distributed across spatial dimensions without spatial correlation.
- **Structured Noise**: Vertical striping visible in individual band images due to detector calibration residuals.
- **Severe Degradation**: Certain bands (e.g., at edges) show extremely high variance and low signal, indicating dead or saturated pixels.

## Link Noise to Environment
- **CRISM**: Experiences low SNR due to Martian atmospheric dust and lower illumination at certain geometries. Instrument temperature variations also contribute to structured noise.

## Justification for Hybrid Preprocessing
A single technique is insufficient. Physical corrections (radiometric/atmospheric) address systematic physics-based variations, while ML-based denoising (CNNs, Autoencoders) are needed to remove the complex structured striping and random noise without destroying subtle mineral absorption features. This hybrid approach ensures signal fidelity and optimal SNR.
