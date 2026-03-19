# CRISM Dataset Analysis & Physical Correction Report

## Dataset Details
- **Dimensions (Lines, Samples, Bands):** 15 x 64 x 107
- **Data Type:** PC_REAL
- **Interleave:** LINE_INTERLEAVED
- **Total Pixels:** 960

## Detected Noises
- **Low SNR Bands (< 5):** 107 bands identified.
- **Dead/Corrupted Pixels:** 2590 pixels found with invalid/zero values.
- **Striping Magnitude:** 2079.7488 (Indicates vertical column striping).
- **Atmospheric Absorption:** Broad dips present in the spectral profile due to CO2/H2O.

## Physical Correction Techniques Applied

### 1. Radiometric Normalization
- **Method:** Data clipping and scaling.
- **Why it was used:** The raw data contained negative values and dead pixels from sensor anomalies or empty structures. We clamped values to a small positive epsilon (1e-6) to ensure physical constraints for reflectance and atmospheric calculations without introducing mathematical errors (like division by zero or NaN generation).
- **What improved:** It prevented dark-current arithmetic underflow, allowing logarithmic and division-based geometric corrections to run seamlessly.

### 2. Illumination & Photometric Normalization
- **Method:** Cross-track column mean normalization (`Relative Gain` division).
- **Why it was used:** Pushbroom sensors like CRISM exhibit strong vertical striping because each column corresponds to a distinct physical detector pixel on the array, each with slight sensitivity variations.
- **What improved:** Extracting the column mean divided by the global mean produces a highly accurate relative gain map. Dividing the raw cube by this matrix dramatically neutralizes vertical striping artifacts in the spatial domain without using complex interpolative AI (CNNs).

### 3. Atmospheric Correction
- **Method:** Internal Average Relative Reflectance (IARR).
- **Why it was used:** The raw radiant energy data includes heavy atmospheric transmittance absorption (from Martian CO2, airborne Dust). IARR calculates a global mean spectrum for the entire scene and divides every pixel by this mean.
- **What improved:** This pure physical ratio suppresses global atmospheric gas absorption features and broad solar irradiance curves. It converts raw Radiance into Relative Reflectance, specifically enhancing and isolating localized mineralogical anomalies rather than atmospheric noise.
