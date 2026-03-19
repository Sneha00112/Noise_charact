import os
import numpy as np
import matplotlib.pyplot as plt

def load_crism(img_path):
    lines, samples, bands = 15, 64, 107
    dtype = np.float32 
    total_elements = lines * samples * bands
    data = np.fromfile(img_path, dtype=dtype, count=total_elements)
    cube = data.reshape((lines, bands, samples))
    cube = np.transpose(cube, (0, 2, 1))
    return cube

if __name__ == "__main__":
    img_path = "/Users/snehasr/Desktop/DRDO/Orig CRISMD2/frt0001073b_01_ra156s_trr3.img"
    cube = load_crism(img_path)
    
    prep_path = "/Users/snehasr/Desktop/DRDO/Orig CRISMD2/Results/preprocessed_cube.npy"
    prep_cube = np.load(prep_path)
    
    out_dir = "/Users/snehasr/Desktop/DRDO/Orig CRISMD2/Results"
    
    # Select a band with visible noise/striping, e.g., band 10 or 50
    b = 50
    raw_band = cube[:, :, b]
    prep_band = prep_cube[:, :, b]
    
    # Normalize for display
    def norm(img):
        return (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
        
    plt.figure(figsize=(15, 5))
    
    # Raw Band
    plt.subplot(1, 3, 1)
    plt.imshow(norm(raw_band), cmap='gray')
    plt.title(f'Raw Band {b} (Original Noise & Striping)')
    plt.colorbar(fraction=0.046, pad=0.04)
    
    # Preprocessed Band
    plt.subplot(1, 3, 2)
    plt.imshow(norm(prep_band), cmap='gray')
    plt.title(f'Preprocessed Band {b} (Cleaned)')
    plt.colorbar(fraction=0.046, pad=0.04)
    
    # Difference
    diff = norm(raw_band) - norm(prep_band)
    plt.subplot(1, 3, 3)
    plt.imshow(np.abs(diff), cmap='hot')
    plt.title('Absolute Difference Map\n(Noise & Artifacts Removed)')
    plt.colorbar(fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'spatial_noise_comparison.png'))
    print("Visual summary saved successfully.")
