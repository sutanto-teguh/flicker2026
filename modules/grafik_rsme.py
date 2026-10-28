import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------
# 1. DATA SIMULASI / INGEST DATA HASIL EKSPERIMEN
# ---------------------------------------------------------
# Urutan bingkai (frame index)
frames = [f"c{i}" for i in [0, 1, 2, 4, 6, 9]]  # Mengikuti urutan Gambar 1 paper Anda
x_axis = np.arange(len(frames))

# Nilai RMSE Ter-normalisasi (0 - 1) per bingkai
rmse_raw = [0.0000, 0.0412, 0.1580, 0.1876, 0.2765, 0.1320]  # SIFT Baseline (Gambar 1)
rmse_ghe = [0.0000, 0.0380, 0.1250, 0.1410, 0.1982, 0.1050]  # Global HE
rmse_clahe = [0.0000, 0.0290, 0.0980, 0.1120, 0.1543, 0.0820]  # CLAHE
rmse_proposed = [0.0000, 0.0080, 0.0210, 0.0280, 0.0421, 0.0190]  # Proposed Adaptive Log-Ch

# ---------------------------------------------------------
# 2. PENGATURAN TAMPILAN GRAFIK (IEEE / JURNAL STYLE)
# ---------------------------------------------------------
plt.figure(figsize=(8, 5), dpi=300)  # High-DPI untuk publikasi
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# Plot Garis untuk Setiap Metode
plt.plot(x_axis, rmse_raw, marker='o', linewidth=2, color='#d62728', label='Raw (No Enhancement)')
plt.plot(x_axis, rmse_ghe, marker='s', linewidth=2, linestyle='--', color='#ff7f0e', label='Global HE')
plt.plot(x_axis, rmse_clahe, marker='^', linewidth=2, linestyle='-.', color='#1f77b4', label='CLAHE')
plt.plot(x_axis, rmse_proposed, marker='D', linewidth=2.5, color='#2ca02c', label='Proposed Adaptive Log-Ch')

# Anotasi Titik Puncak Flicker (c6)
plt.annotate('Peak Flicker (RMSE = 0.2765)', 
             xy=(4, 0.2765), 
             xytext=(2.2, 0.25),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
             fontsize=9, fontweight='bold')

# Kustomisasi Sumbu dan Judul
plt.title('Temporal Flicker Intensity Trend Across Frame Sequence', fontsize=12, fontweight='bold', pad=15)
plt.xlabel('Frame Sequence / Image Index', fontsize=10, fontweight='bold')
plt.ylabel('Normalized Intensity RMSE', fontsize=10, fontweight='bold')
plt.xticks(x_axis, frames, fontsize=10)
plt.yticks(np.arange(0.0, 0.35, 0.05), fontsize=10)
plt.ylim(-0.01, 0.32)

# Grid & Legenda
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9, fontsize=9)

# ---------------------------------------------------------
# 3. SIMPAN DAN TAMPILKAN HASIL
# ---------------------------------------------------------
plt.tight_layout()
plt.savefig('grafik_tren_rmse.png', dpi=300, bbox_inches='tight')
plt.show()
