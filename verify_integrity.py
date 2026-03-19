import numpy as np
import os

def calculate_snr(cube):
    # SNR = mean / std (spatial std per band)
    signal = np.mean(cube, axis=(0,1))
    noise = np.std(cube, axis=(0,1))
    snr_vals = np.where(noise > 0, signal / noise, 0)
    return 20 * np.log10(np.mean(snr_vals) + 1e-6)

def calculate_sam(ref_cube, den_cube):
    # SAM = arccos( dot(X, Y) / (||X|| ||Y||) )
    # Reshape to (pixels, bands)
    ref_flat = ref_cube.reshape(-1, ref_cube.shape[-1])
    den_flat = den_cube.reshape(-1, den_cube.shape[-1])
    
    dot_prod = np.sum(ref_flat * den_flat, axis=1)
    norm_ref = np.linalg.norm(ref_flat, axis=1)
    norm_den = np.linalg.norm(den_flat, axis=1)
    
    cos_theta = dot_prod / (norm_ref * norm_den + 1e-9)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    sam_rad = np.arccos(cos_theta)
    return np.mean(np.degrees(sam_rad))

def calculate_psnr(ref, den):
    mse = np.mean((ref - den) ** 2)
    if mse == 0: return 100
    max_val = np.max(ref)
    return 20 * np.log10(max_val / np.sqrt(mse))

# Paths
ROOT = "/Users/snehasr/Desktop/DRDO/Orig CRISMD2/18:02"
PHYSICAL_PATH = os.path.join(ROOT, "Pipeline_Results/pc_cube.npy")
FINAL_PATH = os.path.join(ROOT, "Advanced_Results/fully_denoised_cube.npy")

print("="*60)
print(" CRISM SPECTRAL INTEGRITY ANALYSIS")
print("="*60)

# Load
phys = np.load(PHYSICAL_PATH)
final = np.load(FINAL_PATH)

# SNR
snr_phys = calculate_snr(phys)
snr_final = calculate_snr(final)
snr_gain = snr_final - snr_phys

# SAM (Spectral Shape Preservation)
mean_sam = calculate_sam(phys, final)

# PSNR (Overall Fidelity)
psnr_val = calculate_psnr(phys, final)

# Feature Preservation (RMSE on normalized spectra)
# Normalize to [0,1] per pixel to focus on shape
phys_norm = phys / (np.max(phys, axis=2, keepdims=True) + 1e-9)
final_norm = final / (np.max(final, axis=2, keepdims=True) + 1e-9)
rmse_shape = np.sqrt(np.mean((phys_norm - final_norm)**2))

print(f"1. SNR Improvement:")
print(f"   - Physical SNR: {snr_phys:.2f} dB")
print(f"   - Final SNR:    {snr_final:.2f} dB")
print(f"   - Boost:        +{snr_gain:.2f} dB")
print()
print(f"2. Spectral Angle Mapper (SAM):")
print(f"   - Mean SAM Error: {mean_sam:.4f} degrees")
print(f"   (Ideally < 0.1 deg for high preservation)")
print()
print(f"3. Feature Preservation Metrics:")
print(f"   - PSNR (vs Physical): {psnr_val:.2f} dB")
print(f"   - Normalized RMSE:    {rmse_shape:.6f}")
print()
print(f"4. Scientific Interpretation:")
if mean_sam < 0.05:
    print("   ✅ SPECTRAL SHAPE PERFECTLY PRESERVED")
else:
    print("   ⚠ MINOR SHAPE DISTORTION DETECTED")

if snr_gain > 0:
    print(f"   ✅ NOISE REDUCTION CONFIRMED (+{snr_gain:.2f} dB)")
else:
    print("   ❌ NOISE INCREASED")

print("="*60)
