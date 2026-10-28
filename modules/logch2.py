import cv2
import numpy as np
import os
import glob

def match_contrast_stats(src, ref):
    """
    Menyesuaikan mean dan stddev citra hasil rekonstruksi (src) 
    agar sesuai dengan citra referensi asli (ref) guna meniadakan 
    peregangan kontras buatan akibat fungsi eksponensial.
    """
    src_mean, src_std = np.mean(src), np.std(src)
    ref_mean, ref_std = np.mean(ref), np.std(ref)
    
    if src_std < 1e-5:
        return np.full_like(src, ref_mean)
        
    matched = ref_mean + (src - src_mean) * (ref_std / src_std)
    return matched

def improved_adaptive_log_chromaticity(image_sequence, window_size=5, spatial_ratio=0.05):
    """
    Mereduksi flicker dengan kombinasi Invariansi Log-Chromaticity,
    Pemulusan Spasial-Temporal 2D, dan Penyelarasan Statistik Kontras.
    """
    num_frames = len(image_sequence)
    if num_frames == 0:
        return []

    h, w = image_sequence[0].shape[:2]
    
    # Tentukan ukuran kernel Gaussian spasial 2D berdasarkan resolusi gambar
    ksize = int(min(h, w) * spatial_ratio)
    if ksize % 2 == 0:
        ksize += 1
    ksize = max(15, ksize)

    is_color = len(image_sequence[0].shape) == 3 and image_sequence[0].shape[2] == 3
    
    log_intensities = []
    log_chromaticities = []
    spatial_illum_maps = []

    # 1. Konversi ke Domain Logaritmik & Isolasi Iluminasi Spasial 2D
    for img in image_sequence:
        img_float = img.astype(np.float64) + 1.0
        log_img = np.log(img_float)

        if is_color:
            # Log-Intensity (Rata-rata Geometris)
            log_intensity = np.mean(log_img, axis=2)
            # Log-Chromaticity: C_c = ln(I_c) - ln(I_geo)
            log_chroma = log_img - log_intensity[:, :, np.newaxis]
            
            log_intensities.append(log_intensity)
            log_chromaticities.append(log_chroma)
        else:
            log_intensity = log_img
            log_intensities.append(log_intensity)

        # Estimasi Peta Iluminasi Lokal 2D (Gaussian Blur pada domain log)
        blur_illum = cv2.GaussianBlur(log_intensity, (ksize, ksize), 0)
        spatial_illum_maps.append(blur_illum)

    spatial_illum_maps = np.array(spatial_illum_maps)  # Shape: (N, H, W)

    # 2. Moving Average Temporal 3D (Spasial-Temporal Smoothing)
    half_win = window_size // 2
    smooth_illum_maps = np.zeros_like(spatial_illum_maps)

    for i in range(num_frames):
        start_idx = max(0, i - half_win)
        end_idx = min(num_frames, i + half_win + 1)
        # Rata-rata temporal untuk setiap titik koordinat (x, y)
        smooth_illum_maps[i] = np.mean(spatial_illum_maps[start_idx:end_idx], axis=0)

    # 3. Koreksi Bias Flicker Lokal 2D & Rekonstruksi Citra
    processed_sequence = []
    for i in range(num_frames):
        # Bias flicker spasial 2D = peta_saat_ini - peta_smooth
        flicker_bias_2d = spatial_illum_maps[i] - smooth_illum_maps[i]
        
        # Koreksi intensitas log per-piksel
        corrected_log_intensity = log_intensities[i] - flicker_bias_2d

        if is_color:
            corrected_log_img = log_chromaticities[i] + corrected_log_intensity[:, :, np.newaxis]
        else:
            corrected_log_img = corrected_log_intensity

        # Rekonstruksi ke domain linear: exp(ln(I)) - 1
        linear_img = np.exp(corrected_log_img) - 1.0

        # 4. Normalisasi Skala Kontras (Mean & Std Matching ke Input Asli)
        orig_img_float = image_sequence[i].astype(np.float64)
        if is_color:
            matched_img = np.zeros_like(linear_img)
            for c in range(3):
                matched_img[:, :, c] = match_contrast_stats(linear_img[:, :, c], orig_img_float[:, :, c])
        else:
            matched_img = match_contrast_stats(linear_img, orig_img_float)

        # Clipping ke batas valid piksel 8-bit
        output_img = np.clip(matched_img, 0, 255).astype(np.uint8)
        processed_sequence.append(output_img)

    return processed_sequence


def process_dataset(scenario_name, input_folder, output_folder, window_size=5):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff',
                  '*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.TIF', '*.TIFF')
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(input_folder, ext)))
    image_paths = sorted(list(set(image_paths)))

    if len(image_paths) == 0:
        print(f"[{scenario_name.upper()}] Peringatan: Tidak ada gambar di '{input_folder}'")
        return

    print(f"[{scenario_name.upper()}] Membaca {len(image_paths)} gambar...")
    
    raw_images = []
    valid_paths = []
    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            if len(img.shape) == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            raw_images.append(img)
            valid_paths.append(path)

    print(f"[{scenario_name.upper()}] Memproses Improved Adaptive Log-Chromaticity...")
    stabilized_images = improved_adaptive_log_chromaticity(raw_images, window_size=window_size)

    for path, img_out in zip(valid_paths, stabilized_images):
        filename = os.path.basename(path)
        cv2.imwrite(os.path.join(output_folder, filename), img_out)

    print(f"[{scenario_name.upper()}] Selesai! Hasil disimpan di: '{output_folder}'\n")


# --- Eksekusi Program ---
base_path = "data_penelitian"
scenarios = ["cctv", "medical", "underwater"]

for scn in scenarios:
    input_dir = os.path.join(base_path, scn, "input")
    output_dir = os.path.join(base_path, scn, "output_logch")
    
    process_dataset(scn, input_dir, output_dir, window_size=5)
