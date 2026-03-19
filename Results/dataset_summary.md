# Dataset Summary Table

| Property | Value |
|---|---|
| Dataset | frt0001073b_01_ra156s_trr3 (CRISM) |
| Dimensions (Lines x Samples x Bands) | 15 x 64 x 107 |
| Spectral Range | VNIR (Visible Near-Infrared) typically 400-1000 nm for S sensor |
| Spatial Resolution | ~18-36 m/pixel depending on altitude |
| Known noise issues | Spikes, striping, low SNR at edges of detector |
| Missing/corrupted bands | Often bands near water absorption or detector edges |

## Noise Behavior Notes
- **CRISM**: Exhibits vertical striping due to column-wise readout inconsistencies.
- **CRISM**: Isolated spikes due to cosmic rays or detector defects.
- **CRISM**: Low SNR in specific bands, especially near limits of the detector (e.g. UV edge or near 1000 nm).
