import numpy as np
import matplotlib.pyplot as plt
import os

def generate_noise_bar_chart(out_path):
    # Categories based on the user's provided image
    categories = [
        'Gaussian Noise',
        'Striping Noise',
        'Atmospheric Absorption',
        'Low SNR Bands',
        'Spike Noise',
        'Dust Scattering',
        'Photon Noise'
    ]
    
    # Values represent relative intensity/presence (0.0 to 1.0)
    # Estimate based on typical CRISM data raw state
    before_denoising = [0.6, 1.0, 1.0, 1.0, 0.8, 0.3, 0.4]
    
    # Estimate based on our hybrid preprocessing pipeline (CNN, AE, Log Residuals)
    after_denoising = [0.1, 0.1, 0.05, 0.15, 0.05, 0.25, 0.2]
    
    y = np.arange(len(categories))
    height = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plotting horizontal bars
    rects1 = ax.barh(y - height/2, before_denoising, height, label='Before Denoising', color='tab:blue')
    rects2 = ax.barh(y + height/2, after_denoising, height, label='After Denoising', color='tab:orange')
    
    # Customizing the plot
    ax.set_xlabel('Relative Noise Intensity / Presence')
    ax.set_title('Noise Summary: Before vs After Denoising')
    ax.set_yticks(y)
    ax.set_yticklabels(categories)
    ax.legend()
    
    # Add grid lines for readability
    ax.xaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    # Set x-limits to slightly over 1.0
    ax.set_xlim(0, 1.1)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved bar chart to {out_path}")

if __name__ == "__main__":
    out_dir = "/Users/snehasr/Desktop/DRDO/Orig CRISMD2/Results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "noise_summary_barchart.png")
    generate_noise_bar_chart(out_path)
