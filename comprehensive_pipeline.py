import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ==========================================================
# 1. LOAD DATA & STRUCTURE ANALYSIS
# ==========================================================
def load_and_analyze(lbl_path, img_path):
    print("--- 1. Data Loading & Analysis ---")
    lines, samples, bands = 15, 64, 107
    dtype = np.float32 
    total_elements = lines * samples * bands
    
    data = np.fromfile(img_path, dtype=dtype, count=total_elements)
    cube = data.reshape((lines, bands, samples))
    cube = np.transpose(cube, (0, 2, 1))
    
    # Scale cube to [0,1] inherently for easier modeling
    cube = np.nan_to_num(cube, nan=1e-6)
    c_min, c_max = np.min(cube), np.max(cube)
    cube_scaled = (cube - c_min) / (c_max - c_min + 1e-6)
    
    print(f"Spatial Dimensions: {lines} Lines x {samples} Samples")
    print(f"Spectral Bands: {bands}")
    print(f"Approx Wavelength Range: 1.0 µm to 3.92 µm (VNIR/SWIR)")
    print(f"Basic Stats -> Min: {np.min(cube_scaled):.4f}, Max: {np.max(cube_scaled):.4f}, Mean: {np.mean(cube_scaled):.4f}, Std: {np.std(cube_scaled):.4f}")
    
    return cube_scaled

# ==========================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================================
def perform_eda(cube, out_dir):
    print("--- 2. Exploratory Data Analysis ---")
    os.makedirs(out_dir, exist_ok=True)
    
    # A. Band-wise variance (Detects noisy bands)
    b_var = np.var(cube, axis=(0,1))
    plt.figure()
    plt.plot(b_var, color='purple')
    plt.title('Band-wise Variance')
    plt.xlabel('Band Index')
    plt.ylabel('Variance')
    plt.savefig(os.path.join(out_dir, 'eda_variance.png'))
    plt.close()
    
    # B. Histogram (Detects data ranges and outlier spikes)
    plt.figure()
    plt.hist(cube.flatten(), bins=100, color='gray', log=True)
    plt.title('Dataset Histogram (Log Scale)')
    plt.xlabel('Reflectance (Normalized)')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(out_dir, 'eda_histogram.png'))
    plt.close()
    
    # C. Spectral Profiles (Detects Atmospheric Drops and random noise)
    plt.figure()
    for i in range(3):
        r, c = np.random.randint(0, cube.shape[0]), np.random.randint(0, cube.shape[1])
        plt.plot(cube[r, c, :], label=f'Pixel({r},{c})', alpha=0.7)
    plt.title('Raw Spectral Profiles')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'eda_spectra.png'))
    plt.close()
    
    # D. Spatial Maps (Detects Striping)
    plt.figure()
    plt.imshow(cube[:, :, 50], cmap='gray')
    plt.title('Band 50 Spatial Map (Highlights Vertical Striping)')
    plt.colorbar()
    plt.savefig(os.path.join(out_dir, 'eda_spatial_map.png'))
    plt.close()

def calc_snr(cube):
    """Calculates global SNR (Mean / Std over spatial dimensions per band, averaged)"""
    b_mean = np.mean(cube, axis=(0,1))
    b_std = np.std(cube, axis=(0,1))
    snr = np.where(b_std > 1e-6, b_mean / b_std, 1e-6)
    return np.mean(snr)

# ==========================================================
# 3. SELECTIVE PREPROCESSING (PHYSICAL)
# ==========================================================
def physical_corrections(cube, snr_tracking):
    print("--- 3.1 Physical Corrections ---")
    
    # 1. Radiometric Normalization
    mins = np.min(cube, axis=(0, 1), keepdims=True)
    maxs = np.max(cube, axis=(0, 1), keepdims=True)
    r_cube = (cube - mins) / (maxs - mins + 1e-6)
    snr_tracking['1. Radiometric Normalization'] = calc_snr(r_cube)
    
    # 2. Illumination Normalization (Addresses vertical striping & scene shading)
    c_means = np.mean(r_cube, axis=0, keepdims=True)
    c_means[c_means == 0] = 1e-6
    i_cube = r_cube / c_means
    i_cube = (i_cube - np.min(i_cube)) / (np.max(i_cube) - np.min(i_cube) + 1e-6)
    snr_tracking['2. Illumination Normalization'] = calc_snr(i_cube)
    
    # 3. Atmospheric Correction Variant (Empirical baseline flattening for structural absorption)
    from scipy.ndimage import uniform_filter1d
    a_cube = uniform_filter1d(i_cube, size=3, axis=2) # Smooth spectral drops physically
    snr_tracking['3. Atmospheric Correction'] = calc_snr(a_cube)
    
    return a_cube, snr_tracking

# ==========================================================
# 4. SELECTIVE PREPROCESSING (ML DENOISING)
# ==========================================================
class Autoencoder(nn.Module):
    def __init__(self, bands):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(bands, 64), nn.ReLU(), nn.Linear(64, 32))
        self.dec = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, bands), nn.Sigmoid())
    def forward(self, x):
        return self.dec(self.enc(x))

class SpatialCNN(nn.Module):
    def __init__(self, bands):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(bands, bands, 3, padding=1, groups=bands), 
            nn.ReLU(),
            nn.Conv1d(bands, bands, 3, padding=1, groups=bands),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.conv(x)

def ml_denoising(cube, snr_tracking):
    print("--- 3.2 ML-Based Denoising ---")
    lines, samples, bands = cube.shape
    
    # 1. Autoencoder (Random Noise Reduction)
    print("Training Autoencoder...")
    pixels = cube.reshape(-1, bands)
    tx = torch.Tensor(pixels)
    loader = DataLoader(TensorDataset(tx, tx), batch_size=128, shuffle=True)
    
    ae = Autoencoder(bands)
    opt = optim.Adam(ae.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    
    for _ in range(40):
        for b_x, _ in loader:
            opt.zero_grad()
            loss = loss_fn(ae(b_x), b_x)
            loss.backward()
            opt.step()
            
    with torch.no_grad():
        ae_cube = ae(tx).numpy().reshape(lines, samples, bands)
    
    # Normalize 0-1
    ae_cube = (ae_cube - np.min(ae_cube)) / (np.max(ae_cube) - np.min(ae_cube) + 1e-6)
    snr_tracking['4. Autoencoder (Random Noise)'] = calc_snr(ae_cube)
    
    # 2. CNN (Striping & Residual Spatial Filtering)
    print("Training CNN...")
    cnn = SpatialCNN(bands)
    opt_cnn = optim.Adam(cnn.parameters(), lr=0.01)
    
    cnn_in = np.transpose(ae_cube, (0, 2, 1)) # (Lines, Bands, Samples)
    t_img = torch.Tensor(cnn_in)
    
    for _ in range(30):
        opt_cnn.zero_grad()
        out = cnn(t_img)
        tv_loss = torch.mean(torch.abs(out[:, :, :-1] - out[:, :, 1:])) * 0.5 # TV loss
        loss = loss_fn(out, t_img) + tv_loss
        loss.backward()
        opt_cnn.step()
        
    with torch.no_grad():
        cnn_out = cnn(t_img).numpy()
    
    final_cube = np.transpose(cnn_out, (0, 2, 1))
    final_cube = (final_cube - np.min(final_cube)) / (np.max(final_cube) - np.min(final_cube) + 1e-6)
    snr_tracking['5. CNN (Spatial Striping)'] = calc_snr(final_cube)
    
    return final_cube, snr_tracking

# ==========================================================
# 5. VISUALIZATIONS & REPORTING
# ==========================================================
def finalize_outputs(raw_cube, final_cube, snr_tracking, out_dir):
    print("--- 4. Finalizing Outputs ---")
    
    # A. SNR Progress Bar Chart
    steps = list(snr_tracking.keys())
    snrs = list(snr_tracking.values())
    
    plt.figure(figsize=(10, 6))
    plt.barh(steps, snrs, color='tab:green')
    plt.title('Progressive SNR Improvement Tracking')
    plt.xlabel('Global Signal-to-Noise Ratio (Higher = Better)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'snr_progression.png'))
    plt.close()
    
    # B. Feature Preservation (Spectra Comparison)
    r, c = 7, 32
    plt.figure(figsize=(10, 5))
    plt.plot(raw_cube[r, c, :], 'r-', alpha=0.5, label='Raw Spectra')
    plt.plot(final_cube[r, c, :], 'b-', linewidth=2, label='Fully Denoised Spectra')
    plt.title(f'Absorption Feature Preservation (Pixel {r},{c})')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'feature_preservation.png'))
    plt.close()
    
    # C. Spatial Comparison
    b = 50
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(raw_cube[:, :, b], cmap='gray')
    plt.title(f'Raw Spatial Map (Band {b})')
    plt.subplot(1, 2, 2)
    plt.imshow(final_cube[:, :, b], cmap='gray')
    plt.title(f'Denoised Spatial Map (Band {b})')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'spatial_comparison.png'))
    plt.close()
    
    # D. Datasets
    np.save(os.path.join(out_dir, 'clean_dataset.npy'), final_cube)
    
    # E. Construct output markdown report lines (to be copied to System Artifact subsequently)
    print("Writing text report details...")

if __name__ == "__main__":
    out_dir = "/Users/snehasr/Desktop/DRDO/Orig CRISMD2/Comprehensive_Results"
    lbl = "/Users/snehasr/Desktop/DRDO/Orig CRISMD2/frt0001073b_01_ra156s_trr3.lbl.txt"
    img = "/Users/snehasr/Desktop/DRDO/Orig CRISMD2/frt0001073b_01_ra156s_trr3.img"
    
    raw_cube = load_and_analyze(lbl, img)
    perform_eda(raw_cube, out_dir)
    
    snr_tracking = {'0. Raw Dataset Baseline': calc_snr(raw_cube)}
    
    pc_cube, snr_tracking = physical_corrections(raw_cube, snr_tracking)
    ml_cube, snr_tracking = ml_denoising(pc_cube, snr_tracking)
    
    finalize_outputs(raw_cube, ml_cube, snr_tracking, out_dir)
    print("Pipeline Execution Complete. Images saved in Comprehensive_Results.")
