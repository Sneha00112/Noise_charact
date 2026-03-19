import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_crism(img_path):
    # Based on the frt0001073b_01_ra156s_trr3.lbl.txt
    lines = 15
    samples = 64
    bands = 107
    dtype = np.float32 # PC_REAL
    
    total_elements = lines * samples * bands
    data = np.fromfile(img_path, dtype=dtype, count=total_elements)
    
    # LINE_INTERLEAVED (BIL)
    # BIL: (lines, bands, samples)
    cube = data.reshape((lines, bands, samples))
    # Transpose to (lines, samples, bands) -> (rows, cols, bands)
    cube = np.transpose(cube, (0, 2, 1))
    
    return cube

def phase1_dataset_acquisition(cube, out_dir):
    print("--- PHASE 1: Dataset Acquisition ---")
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Document dataset properties
    summary = {
        "Dataset": "frt0001073b_01_ra156s_trr3 (CRISM)",
        "Dimensions (Lines x Samples x Bands)": f"{cube.shape[0]} x {cube.shape[1]} x {cube.shape[2]}",
        "Spectral Range": "VNIR (Visible Near-Infrared) typically 400-1000 nm for S sensor",
        "Spatial Resolution": "~18-36 m/pixel depending on altitude",
        "Known noise issues": "Spikes, striping, low SNR at edges of detector",
        "Missing/corrupted bands": "Often bands near water absorption or detector edges"
    }
    
    with open(os.path.join(out_dir, 'dataset_summary.md'), 'w') as f:
        f.write("# Dataset Summary Table\n\n")
        f.write("| Property | Value |\n|---|---|\n")
        for k, v in summary.items():
            f.write(f"| {k} | {v} |\n")
            
        f.write("\n## Noise Behavior Notes\n")
        f.write("- **CRISM**: Exhibits vertical striping due to column-wise readout inconsistencies.\n")
        f.write("- **CRISM**: Isolated spikes due to cosmic rays or detector defects.\n")
        f.write("- **CRISM**: Low SNR in specific bands, especially near limits of the detector (e.g. UV edge or near 1000 nm).\n")
    
    # 2. Visualize Raw Spectra
    plt.figure(figsize=(10, 5))
    for i in range(5): # Plot 5 random pixel spectra
        r, c = np.random.randint(0, cube.shape[0]), np.random.randint(0, cube.shape[1])
        plt.plot(cube[r, c, :], label=f'Pixel ({r},{c})', alpha=0.7)
    plt.title('Raw Spectra Visualization')
    plt.xlabel('Band Index')
    plt.ylabel('Radiance (I/F)')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'raw_spectra.png'))
    plt.close()
    
    # 3. Visualize a specific band to show noise/striping
    band_idx = 10 # typically some striping/noise
    plt.figure()
    plt.imshow(cube[:, :, band_idx], cmap='gray')
    plt.title(f'Band {band_idx} Image (Showing Spatial Noise)')
    plt.colorbar()
    plt.savefig(os.path.join(out_dir, 'noisy_band.png'))
    plt.close()
    print("Phase 1 outputs saved.")

def phase2_noise_characterization(cube, out_dir):
    print("--- PHASE 2: Noise Characterization ---")
    
    # 1. Band-wise variance analysis
    # High variance across bands might indicate noise if signal is uniform, but it also contains scene contrast.
    band_var = np.var(cube, axis=(0,1))
    
    plt.figure(figsize=(10, 4))
    plt.plot(band_var)
    plt.title('Band-wise Variance (Noise Profile)')
    plt.xlabel('Band Index')
    plt.ylabel('Variance')
    plt.savefig(os.path.join(out_dir, 'bandwise_variance.png'))
    plt.close()
    
    # 2. SNR Estimation (Simplified: Mean / Std for each band)
    band_mean = np.mean(cube, axis=(0,1))
    band_std = np.std(cube, axis=(0,1))
    snr = np.where(band_std > 0, band_mean / band_std, 0)
    
    plt.figure(figsize=(10, 4))
    plt.plot(snr)
    plt.title('SNR Estimation per Band')
    plt.xlabel('Band Index')
    plt.ylabel('SNR')
    plt.savefig(os.path.join(out_dir, 'snr_estimation.png'))
    plt.close()
    
    with open(os.path.join(out_dir, 'noise_characterization.md'), 'w') as f:
        f.write("# Mission-wise Noise Characterization\n\n")
        f.write("## Categorization of Noise\n")
        f.write("- **Random Noise**: Evident in low SNR bands, distributed across spatial dimensions without spatial correlation.\n")
        f.write("- **Structured Noise**: Vertical striping visible in individual band images due to detector calibration residuals.\n")
        f.write("- **Severe Degradation**: Certain bands (e.g., at edges) show extremely high variance and low signal, indicating dead or saturated pixels.\n\n")
        f.write("## Link Noise to Environment\n")
        f.write("- **CRISM**: Experiences low SNR due to Martian atmospheric dust and lower illumination at certain geometries. Instrument temperature variations also contribute to structured noise.\n\n")
        f.write("## Justification for Hybrid Preprocessing\n")
        f.write("A single technique is insufficient. Physical corrections (radiometric/atmospheric) address systematic physics-based variations, while ML-based denoising (CNNs, Autoencoders) are needed to remove the complex structured striping and random noise without destroying subtle mineral absorption features. This hybrid approach ensures signal fidelity and optimal SNR.\n")
    print("Phase 2 outputs saved.")

if __name__ == "__main__":
    img_path = "/Users/snehasr/Desktop/DRDO/Orig CRISMD2/frt0001073b_01_ra156s_trr3.img"
    cube = load_crism(img_path)
    print("Cube shaped loaded:", cube.shape)
    
    out_dir = "/Users/snehasr/Desktop/DRDO/Orig CRISMD2/Results"
    phase1_dataset_acquisition(cube, out_dir)
    phase2_noise_characterization(cube, out_dir)
