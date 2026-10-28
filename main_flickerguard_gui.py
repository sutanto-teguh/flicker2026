"""
================================================================================
MultiDim-FlickerGuard - Master Controller & GUI Engine
Deskripsi: Controller terintegrasi berbasis Graphical User Interface (GUI Tkinter)
           untuk menjalankan pipeline pemrosesan citra dari modul-modul penelitian HKI:
           - Prapemrosesan Flicker (Proposed Log-Ch, CLAHE, GHE)
           - Ekstraksi Fitur & Registrasi (SIFT + FLANN + RANSAC)
           - Evaluasi Kuantitatif (Temporal RMSE, Inliers, Precision Ratio, SSIM)
           - Visualisasi Grafik Tren dan Perbandingan Gambar
================================================================================
"""

import sys
import os
import glob
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd

# ==============================================================================
# 1. SUBSISTEM PRAPEMROSESAN & TRANSFORMASI RUANG WARNA
# ==============================================================================

def match_contrast_stats(src, ref):
    """Menyelaraskan mean dan stddev hasil rekonstruksi dengan citra asli."""
    src_mean, src_std = np.mean(src), np.std(src)
    ref_mean, ref_std = np.mean(ref), np.std(ref)
    if src_std < 1e-5:
        return np.full_like(src, ref_mean)
    return ref_mean + (src - src_mean) * (ref_std / src_std)

def apply_improved_log_chromaticity(image_sequence, window_size=5, spatial_ratio=0.05):
    """Proposed Adaptive Log-Chrominance Flicker Reduction."""
    num_frames = len(image_sequence)
    if num_frames == 0:
        return []
        
    h, w = image_sequence[0].shape[:2]
    ksize = int(min(h, w) * spatial_ratio)
    if ksize % 2 == 0:
        ksize += 1
    ksize = max(15, ksize)
    
    is_color = (len(image_sequence[0].shape) == 3 and image_sequence[0].shape[2] == 3)
    log_intensities = []
    log_chromaticities = []
    spatial_illum_maps = []
    
    # 1. Domain Logaritmik & Peta Iluminasi 2D
    for img in image_sequence:
        img_float = img.astype(np.float64) + 1.0
        log_img = np.log(img_float)
        if is_color:
            log_intensity = np.mean(log_img, axis=2)
            log_chroma = log_img - log_intensity[:, :, np.newaxis]
            log_intensities.append(log_intensity)
            log_chromaticities.append(log_chroma)
        else:
            log_intensity = log_img
            log_intensities.append(log_intensity)
        blur_illum = cv2.GaussianBlur(log_intensity, (ksize, ksize), 0)
        spatial_illum_maps.append(blur_illum)
        
    spatial_illum_maps = np.array(spatial_illum_maps)
    
    # 2. Moving Average Temporal 3D
    half_win = window_size // 2
    smooth_illum_maps = np.zeros_like(spatial_illum_maps)
    for i in range(num_frames):
        st = max(0, i - half_win)
        en = min(num_frames, i + half_win + 1)
        smooth_illum_maps[i] = np.mean(spatial_illum_maps[st:en], axis=0)
        
    # 3. Koreksi Bias Flicker & Rekonstruksi
    processed = []
    for i in range(num_frames):
        bias = spatial_illum_maps[i] - smooth_illum_maps[i]
        corrected_log_intensity = log_intensities[i] - bias
        if is_color:
            corrected_log_img = log_chromaticities[i] + corrected_log_intensity[:, :, np.newaxis]
        else:
            corrected_log_img = corrected_log_intensity
        linear_img = np.exp(corrected_log_img) - 1.0
        
        # 4. Statistical Contrast Matching
        orig_float = image_sequence[i].astype(np.float64)
        if is_color:
            matched_img = np.zeros_like(linear_img)
            for c in range(3):
                matched_img[:, :, c] = match_contrast_stats(linear_img[:, :, c], orig_float[:, :, c])
        else:
            matched_img = match_contrast_stats(linear_img, orig_float)
            
        out_img = np.clip(matched_img, 0, 255).astype(np.uint8)
        processed.append(out_img)
        
    return processed

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Penerapan CLAHE terisolasi pada kanal L (LAB) atau Grayscale."""
    clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    if len(image.shape) == 2 or image.shape[2] == 1:
        return clahe_obj.apply(image)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_c = clahe_obj.apply(l)
    return cv2.cvtColor(cv2.merge((l_c, a, b)), cv2.COLOR_LAB2BGR)

def apply_global_he(image):
    """Penerapan Global Histogram Equalization pada kanal Y (YCrCb) atau Grayscale."""
    if len(image.shape) == 2 or image.shape[2] == 1:
        return cv2.equalizeHist(image)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    return cv2.cvtColor(cv2.merge((y_eq, cr, cb)), cv2.COLOR_YCrCb2BGR)

# ==============================================================================
# 2. SUBSISTEM EVALUASI, METRIK REGISTRASI & TEMPORAL RMSE
# ==============================================================================

def compute_temporal_rmse(img_list):
    """Menghitung Temporal RMSE antar-frame berurutan."""
    if len(img_list) < 2:
        return 0.0, 0.0, []
    rmse_list = []
    for i in range(len(img_list) - 1):
        g1 = cv2.cvtColor(img_list[i], cv2.COLOR_BGR2GRAY) if len(img_list[i].shape) == 3 else img_list[i]
        g2 = cv2.cvtColor(img_list[i+1], cv2.COLOR_BGR2GRAY) if len(img_list[i+1].shape) == 3 else img_list[i+1]
        if g1.shape != g2.shape:
            g2 = cv2.resize(g2, (g1.shape[1], g1.shape[0]))
        mse = np.mean((g1.astype(float) - g2.astype(float)) ** 2)
        rmse_list.append(np.sqrt(mse))
    avg_rmse = float(np.mean(rmse_list))
    norm_rmse = avg_rmse / 255.0
    return avg_rmse, norm_rmse, rmse_list

def evaluate_features_and_ransac(img1, img2, ratio_thresh=0.75, ransac_thresh=5.0):
    """Mengekstrak fitur SIFT, mencocokkan FLANN, dan menapis inliers via RANSAC."""
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
    
    sift = cv2.SIFT_create(nfeatures=3000)
    kp1, des1 = sift.detectAndCompute(g1, None)
    kp2, des2 = sift.detectAndCompute(g2, None)
    
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return 0, 0, 0.0, None
        
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
            
    if len(good_matches) < 4:
        return len(kp1), 0, 0.0, None
        
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
    inliers = int(np.sum(mask)) if mask is not None else 0
    precision = (inliers / len(good_matches)) if len(good_matches) > 0 else 0.0
    
    # Gambar visualisasi pencocokan
    inlier_matches = [good_matches[i] for i in range(len(good_matches)) if mask[i]] if mask is not None else []
    draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=None, flags=2)
    vis_img = cv2.drawMatches(img1, kp1, img2, kp2, inlier_matches, None, **draw_params)
    
    return len(kp1), inliers, precision, vis_img

# ==============================================================================
# 3. MASTER GUI CONTROLLER (TKINTER)
# ==============================================================================

class MultiDimFlickerGuardGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("MultiDim-FlickerGuard: Controller & Dashboard Penelitian HKI")
        self.root.geometry("1180x780")
        self.root.minsize(980, 680)

        # State data
        self.dataset_dir = tk.StringVar(value="data_penelitian")
        self.scenario = tk.StringVar(value="cctv")
        self.raw_images = []
        self.raw_paths = []
        self.processed_dict = {} # 'GHE', 'CLAHE', 'Proposed Log-Ch'
        self.evaluation_results = {}
        
        self.setup_ui()

    def setup_ui(self):
        # Header Banner
        header_frame = tk.Frame(self.root, bg="#1A365D", height=65)
        header_frame.pack(fill=tk.X, side=tk.TOP)
        title_lbl = tk.Label(header_frame, text="MultiDim-FlickerGuard", font=("Helvetica", 16, "bold"), fg="#FFFFFF", bg="#1A365D")
        title_lbl.pack(anchor="w", padx=20, pady=(8, 2))
        sub_lbl = tk.Label(header_frame, text="Penstabil Fluktuasi Luminans & Registrasi Citra Multidomain (CCTV, Medis, Underwater)",
                           font=("Helvetica", 10), fg="#CBD5E0", bg="#1A365D")
        sub_lbl.pack(anchor="w", padx=20, pady=(0, 8))

        # Main Layout (Paned / Split Frame)
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left Control Panel
        left_frame = ttk.Frame(main_paned, width=360)
        main_paned.add(left_frame, weight=0)

        # 1. Dataset & Skenario Section
        grp_data = ttk.LabelFrame(left_frame, text=" 1. Masukan Data (Data Ingestion Layer) ", padding=10)
        grp_data.pack(fill=tk.X, pady=5)

        ttk.Label(grp_data, text="Basis Direktori:").grid(row=0, column=0, sticky="w", pady=2)
        txt_path = ttk.Entry(grp_data, textvariable=self.dataset_dir, width=24)
        txt_path.grid(row=0, column=1, padx=4, pady=2)
        btn_browse = ttk.Button(grp_data, text="Pilih...", width=7, command=self.browse_dir)
        btn_browse.grid(row=0, column=2, pady=2)

        ttk.Label(grp_data, text="Skenario Domain:").grid(row=1, column=0, sticky="w", pady=4)
        cmb_scenario = ttk.Combobox(grp_data, textvariable=self.scenario, values=["cctv", "medical", "underwater"], state="readonly", width=12)
        cmb_scenario.grid(row=1, column=1, sticky="w", padx=4, pady=4)

        btn_load = ttk.Button(grp_data, text="Load Sekuens Citra", command=self.load_images_threaded)
        btn_load.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(6, 2))

        self.lbl_load_status = ttk.Label(grp_data, text="Status: Siap memuat data", foreground="#4A5568", font=("Helvetica", 8))
        self.lbl_load_status.grid(row=3, column=0, columnspan=3, sticky="w", pady=2)

        # 2. Pipeline Execution Section
        grp_pipeline = ttk.LabelFrame(left_frame, text=" 2. Eksekusi Pipeline (Pre-Processing & Evaluation) ", padding=10)
        grp_pipeline.pack(fill=tk.X, pady=5)

        self.btn_run_all = ttk.Button(grp_pipeline, text="Jalankan Semua Pipeline (Raw vs GHE vs CLAHE vs Log-Ch)", command=self.run_pipeline_threaded)
        self.btn_run_all.pack(fill=tk.X, pady=4)

        self.progress_bar = ttk.Progressbar(grp_pipeline, mode="indeterminate")
        self.progress_bar.pack(fill=tk.X, pady=4)

        # 3. Log Output Console
        grp_log = ttk.LabelFrame(left_frame, text=" 3. Konsol Aktivitas & Log Sistem ", padding=10)
        grp_log.pack(fill=tk.BOTH, expand=True, pady=5)

        self.txt_log = tk.Text(grp_log, height=12, wrap=tk.WORD, font=("Consolas", 8), bg="#2D3748", fg="#F7FAFC")
        self.txt_log.pack(fill=tk.BOTH, expand=True)

        # Right Display Area (Notebook Tabs)
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=1)

        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Tabel Metrik Kuantitatif
        self.tab_table = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_table, text="Tabel Ringkasan Metrik (RMSE & Inliers)")
        self.setup_table_tab()

        # Tab 2: Visualisasi Grafik Tren RMSE
        self.tab_plot = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_plot, text="Grafik Tren Temporal RMSE")
        self.setup_plot_tab()

        # Tab 3: Visualisasi Grid Matching
        self.tab_vis = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_vis, text="Visualisasi Grid Pencocokan Fitur")
        self.setup_vis_tab()

    def log(self, text):
        self.txt_log.insert(tk.END, text + "\n")
        self.txt_log.see(tk.END)

    def browse_dir(self):
        d = filedialog.askdirectory(initialdir=os.getcwd(), title="Pilih Folder Direktori Penelitian")
        if d:
            self.dataset_dir.set(d)

    def setup_table_tab(self):
        cols = ("Metode", "Avg RMSE (0-255)", "Norm RMSE (0-1)", "Total Keypoints", "Valid Inliers", "Precision Ratio")
        self.tree = ttk.Treeview(self.tab_table, columns=cols, show="headings", height=8)
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, anchor="center", width=125)
        self.tree.pack(fill=tk.BOTH, expand=True, pady=5)

        btn_save_csv = ttk.Button(self.tab_table, text="Simpan Hasil ke CSV", command=self.save_csv)
        btn_save_csv.pack(anchor="e", pady=5)

    def setup_plot_tab(self):
        self.fig_plot, self.ax_plot = plt.subplots(figsize=(6, 4), dpi=100)
        self.canvas_plot = FigureCanvasTkAgg(self.fig_plot, master=self.tab_plot)
        self.canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def setup_vis_tab(self):
        self.fig_vis, self.ax_vis = plt.subplots(figsize=(7, 5), dpi=100)
        self.canvas_vis = FigureCanvasTkAgg(self.fig_vis, master=self.tab_vis)
        self.canvas_vis.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def load_images_threaded(self):
        threading.Thread(target=self.load_images, daemon=True).start()

    def load_images(self):
        base = self.dataset_dir.get()
        scn = self.scenario.get()
        inp_dir = os.path.join(base, scn, "input")
        
        self.log(f"[*] Mencari citra di folder: {inp_dir}")
        exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff',
                '*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.TIF', '*.TIFF')
        files = []
        for e in exts:
            files.extend(glob.glob(os.path.join(inp_dir, e)))
        files = sorted(list(set(files)))
        
        if not files:
            messagebox.showwarning("Peringatan", f"Tidak ada berkas citra ditemukan di '{inp_dir}'.")
            self.lbl_load_status.config(text="Status: Gagal, berkas tidak ditemukan", foreground="red")
            return
            
        self.raw_images = []
        self.raw_paths = []
        for f in files:
            img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
            if img is not None:
                # Standarisasi kedalaman kanal (RGBA -> BGR)
                if len(img.shape) == 3 and img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                self.raw_images.append(img)
                self.raw_paths.append(f)
                
        self.lbl_load_status.config(text=f"Status: Terbaca {len(self.raw_images)} frame ({scn})", foreground="green")
        self.log(f"[✓] Berhasil memuat {len(self.raw_images)} frame citra untuk domain {scn.upper()}.")

    def run_pipeline_threaded(self):
        if not self.raw_images:
            messagebox.showerror("Eror", "Silakan klik 'Load Sekuens Citra' terlebih dahulu!")
            return
        threading.Thread(target=self.run_pipeline, daemon=True).start()

    def run_pipeline(self):
        self.progress_bar.start(10)
        self.btn_run_all.config(state="disabled")
        
        try:
            base = self.dataset_dir.get()
            scn = self.scenario.get()
            
            # Buat folder output
            out_ghe_dir = os.path.join(base, scn, "output_ghe")
            out_clahe_dir = os.path.join(base, scn, "output_clahe")
            out_logch_dir = os.path.join(base, scn, "output_logch")
            for d in [out_ghe_dir, out_clahe_dir, out_logch_dir]:
                os.makedirs(d, exist_ok=True)
                
            # 1. Prapemrosesan GHE
            self.log("[1/4] Memproses Global Histogram Equalization (GHE)...")
            ghe_images = [apply_global_he(img) for img in self.raw_images]
            for p, img in zip(self.raw_paths, ghe_images):
                cv2.imwrite(os.path.join(out_ghe_dir, os.path.basename(p)), img)
                
            # 2. Prapemrosesan CLAHE
            self.log("[2/4] Memproses Contrast Limited Adaptive HE (CLAHE)...")
            clahe_images = [apply_clahe(img, clip_limit=2.0, tile_grid_size=(8,8)) for img in self.raw_images]
            for p, img in zip(self.raw_paths, clahe_images):
                cv2.imwrite(os.path.join(out_clahe_dir, os.path.basename(p)), img)
                
            # 3. Prapemrosesan Proposed Adaptive Log-Chrominance
            self.log("[3/4] Memproses Proposed Adaptive Log-Chrominance...")
            logch_images = apply_improved_log_chromaticity(self.raw_images, window_size=5)
            for p, img in zip(self.raw_paths, logch_images):
                cv2.imwrite(os.path.join(out_logch_dir, os.path.basename(p)), img)
                
            self.processed_dict = {
                "Raw Input": self.raw_images,
                "Global HE": ghe_images,
                "CLAHE": clahe_images,
                "Proposed Log-Ch": logch_images
            }
            
            # 4. Evaluasi Metrik
            self.log("[4/4] Menghitung Temporal RMSE dan Registrasi SIFT-RANSAC...")
            self.evaluation_results = {}
            for item in self.tree.get_children():
                self.tree.delete(item)
                
            matching_vis_dict = {}
            
            for m_name, seq in self.processed_dict.items():
                avg_rmse, norm_rmse, rmse_seq = compute_temporal_rmse(seq)
                
                # Registrasi pada frame 0 dan frame 1 (atau frame tengah)
                kps, inliers, prec, vis_img = 0, 0, 0.0, None
                if len(seq) >= 2:
                    kps, inliers, prec, vis_img = evaluate_features_and_ransac(seq[0], seq[1])
                matching_vis_dict[m_name] = vis_img
                
                self.evaluation_results[m_name] = {
                    "avg_rmse": avg_rmse,
                    "norm_rmse": norm_rmse,
                    "rmse_seq": rmse_seq,
                    "kps": kps,
                    "inliers": inliers,
                    "precision": prec
                }
                
                # Tambah ke Treeview
                self.tree.insert("", "end", values=(
                    m_name,
                    f"{avg_rmse:.4f}",
                    f"{norm_rmse:.4f}",
                    kps,
                    inliers,
                    f"{prec*100:.2f}%"
                ))
                
            # Render Grafik
            self.render_rmse_plot()
            self.render_vis_grid(matching_vis_dict)
            self.log("[✓] Seluruh pipeline selesai dieksekusi dengan sukses!")
            messagebox.showinfo("Selesai", f"Pipeline komparasi domain '{scn.upper()}' selesai diproses.")
            
        except Exception as e:
            self.log(f"[!] Terjadi Eror: {str(e)}")
            messagebox.showerror("Eror Pipeline", str(e))
        finally:
            self.progress_bar.stop()
            self.btn_run_all.config(state="normal")

    def render_rmse_plot(self):
        self.ax_plot.clear()
        colors = {"Raw Input": "#d62728", "Global HE": "#ff7f0e", "CLAHE": "#1f77b4", "Proposed Log-Ch": "#2ca02c"}
        markers = {"Raw Input": "o", "Global HE": "s", "CLAHE": "^", "Proposed Log-Ch": "D"}
        
        for m_name, data in self.evaluation_results.items():
            seq = [0.0] + [val / 255.0 for val in data["rmse_seq"]]
            x = list(range(len(seq)))
            self.ax_plot.plot(x, seq, marker=markers.get(m_name, "o"), label=m_name, color=colors.get(m_name, "blue"), linewidth=2)
            
        self.ax_plot.set_title(f"Temporal Intensity RMSE Trend ({self.scenario.get().upper()})", fontsize=11, fontweight="bold")
        self.ax_plot.set_xlabel("Urutan Transisi Bingkai (t -> t+1)")
        self.ax_plot.set_ylabel("Normalized Intensity RMSE")
        self.ax_plot.grid(True, linestyle=":", alpha=0.6)
        self.ax_plot.legend()
        self.fig_plot.tight_layout()
        self.canvas_plot.draw()

    def render_vis_grid(self, vis_dict):
        self.ax_vis.clear()
        # Buat Grid 2x2 komparasi
        grid_rows, grid_cols = 2, 2
        names = ["Raw Input", "Global HE", "CLAHE", "Proposed Log-Ch"]
        
        # Gambar gabungan pada kanvas matplotlib
        self.fig_vis.clf()
        for idx, name in enumerate(names):
            ax = self.fig_vis.add_subplot(2, 2, idx + 1)
            img = vis_dict.get(name)
            if img is not None:
                # OpenCV BGR ke RGB untuk Matplotlib
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(rgb)
            ax.set_title(name, fontsize=9, fontweight="bold")
            ax.axis("off")
        self.fig_vis.tight_layout()
        self.canvas_vis.draw()

    def save_csv(self):
        if not self.evaluation_results:
            messagebox.showwarning("Peringatan", "Belum ada data evaluasi untuk disimpan.")
            return
        fp = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
        if fp:
            records = []
            for k, v in self.evaluation_results.items():
                records.append({
                    "Skenario": self.scenario.get(),
                    "Metode": k,
                    "Avg_RMSE_255": v["avg_rmse"],
                    "Norm_RMSE_0_1": v["norm_rmse"],
                    "Keypoints": v["kps"],
                    "Inliers": v["inliers"],
                    "Precision_Ratio": v["precision"]
                })
            df = pd.DataFrame(records)
            df.to_csv(fp, index=False)
            messagebox.showinfo("Tersimpan", f"Tabel ringkasan berhasil disimpan di:\n{fp}")

# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = MultiDimFlickerGuardGUI(root)
    root.mainloop()
