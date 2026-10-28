import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def calculate_temporal_rmse(folder_path):
    """Menghitung rata-rata RMSE temporal antar-bingkai berurutan dalam folder."""
    extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff', '*.JPG', '*.JPEG', '*.PNG', '*.BMP')
    img_files = []
    for ext in extensions:
        img_files.extend(glob.glob(os.path.join(folder_path, ext)))
    img_files = sorted(list(set(img_files)))

    if len(img_files) < 2:
        return []

    rmse_values = []
    for i in range(len(img_files) - 1):
        img1 = cv2.imread(img_files[i], cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img_files[i+1], cv2.IMREAD_GRAYSCALE)

        if img1 is None or img2 is None:
            continue
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        # Hitung MSE dan RMSE dinormalisasi ke skala 0 - 1 (dibagi 255)
        mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
        rmse = np.sqrt(mse) / 255.0
        rmse_values.append(rmse)

    return rmse_values

def plot_rmse_comparison(base_path="data_penelitian"):
    """Membuat grafik perbandingan tren RMSE temporal antara Input, GHE, CLAHE, dan Proposed Log-Ch."""
    scenarios = ["cctv", "medical", "underwater"]
    methods = {
        "Raw (Input)": "input",
        "Global HE": "output_ghe",
        "CLAHE": "output_clahe",
        "Proposed Log-Ch": "output_logch"
    }
    
    styles = {
        "Raw (Input)": {"color": "#d62728", "marker": "o", "linestyle": "-"},
        "Global HE": {"color": "#ff7f0e", "marker": "s", "linestyle": "--"},
        "CLAHE": {"color": "#1f77b4", "marker": "^", "linestyle": "-."},
        "Proposed Log-Ch": {"color": "#2ca02c", "marker": "D", "linestyle": "-"}
    }

    fig, axes = plt.subplots(1, len(scenarios), figsize=(16, 5), dpi=300)
    
    if len(scenarios) == 1:
        axes = [axes]

    for idx, scn in enumerate(scenarios):
        ax = axes[idx]
        has_data = False

        for m_label, m_folder in methods.items():
            folder_path = os.path.join(base_path, scn, m_folder)
            rmse_series = calculate_temporal_rmse(folder_path)

            if rmse_series:
                has_data = True
                x_axis = np.arange(len(rmse_series))
                ax.plot(x_axis, rmse_series, 
                        label=m_label, 
                        color=styles[m_label]["color"], 
                        marker=styles[m_label]["marker"], 
                        linestyle=styles[m_label]["linestyle"], 
                        linewidth=2, markersize=5)

        ax.set_title(f"Skenario: {scn.upper()}", fontsize=11, fontweight='bold', pad=10)
        ax.set_xlabel("Indeks Transisi Frame ($t \\rightarrow t+1$)", fontsize=9, fontweight='bold')
        ax.set_ylabel("Normalized RMSE", fontsize=9, fontweight='bold')
        ax.grid(True, linestyle=":", alpha=0.6)
        
        if has_data:
            ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=8)
        else:
            ax.text(0.5, 0.5, "Data Folder\nTidak Lengkap", ha='center', va='center', transform=ax.transAxes, fontsize=10)

    plt.suptitle("Perbandingan Kinerja Mereduksi Flicker (Temporal RMSE Trend)", fontsize=13, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.show()

# --- Eksekusi Program Grafik ---
plot_rmse_comparison(base_path="data_penelitian")