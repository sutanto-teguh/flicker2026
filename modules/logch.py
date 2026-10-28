import cv2
import numpy as np
import os
import glob

def adaptive_log_chromaticity_reduction(image_sequence, window_size=5):
    """
    Mereduksi flicker sekuensial menggunakan ruang warna Adaptive Log-Chromaticity.
    
    Parameters:
        image_sequence (list): List dari ndarray gambar dalam urutan temporal.
        window_size (int): Ukuran jendela moving-average temporal.
        
    Returns:
        list: List dari ndarray gambar yang telah distabilkan.
    """
    num_frames = len(image_sequence)
    if num_frames == 0:
        return []

    processed_sequence = []
    
    # 1. Konversi ke Domain Logaritmik dan Deteksi Kanal
    is_color = len(image_sequence[0].shape) == 3 and image_sequence[0].shape[2] == 3
    
    log_frames = []
    log_intensities = []
    log_chromaticities = []

    for img in image_sequence:
        # Tambahkan float offset (+1.0) untuk menghindari log(0)
        img_float = img.astype(np.float64) + 1.0
        log_img = np.log(img_float)
        log_frames.append(log_img)

        if is_color:
            # Hitung Log-Intensity (Geometric Mean dari R, G, B)
            log_intensity = np.mean(log_img, axis=2)
            # Hitung Log-Chromaticity: C_c = ln(I_c) - ln(I_geo)
            log_chroma = log_img - log_intensity[:, :, np.newaxis]
            
            log_intensities.append(log_intensity)
            log_chromaticities.append(log_chroma)
        else:
            log_intensities.append(log_img)

    # 2. Estimasi Statistik Temporal (Rata-rata Intensitas Log per Frame)
    frame_means = [np.mean(l_int) for l_int in log_intensities]

    # 3. Moving Average Adaptif untuk Mengisolasi Trend Iluminasi (L)
    half_win = window_size // 2
    smooth_means = []
    for i in range(num_frames):
        start_idx = max(0, i - half_win)
        end_idx = min(num_frames, i + half_win + 1)
        smooth_means.append(np.mean(frame_means[start_idx:end_idx]))

    # 4. Rekonstruksi dan Normalisasi Citra
    for i in range(num_frames):
        # Bias flicker aditif pada domain log = mean_saat_ini - mean_smooth
        flicker_bias = frame_means[i] - smooth_means[i]
        
        # Koreksi intensitas log
        corrected_log_intensity = log_intensities[i] - flicker_bias

        if is_color:
            # Gabungkan kembali Log-Chromaticity dengan Log-Intensity ter-koreksi
            corrected_log_img = log_chromaticities[i] + corrected_log_intensity[:, :, np.newaxis]
        else:
            corrected_log_img = corrected_log_intensity

        # Transformasi kembali ke domain linear: exp(ln(I)) - 1
        linear_img = np.exp(corrected_log_img) - 1.0
        
        # Clipping ke rentang valid [0, 255] dan konversi uint8
        output_img = np.clip(linear_img, 0, 255).astype(np.uint8)
        processed_sequence.append(output_img)

    return processed_sequence


def process_dataset(scenario_name, input_folder, output_folder, window_size=5):
    """
    Membaca, memproses, dan menyimpan sekuens gambar per skenario.
    """
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
    
    # Read frames
    raw_images = []
    valid_paths = []
    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            if len(img.shape) == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            raw_images.append(img)
            valid_paths.append(path)

    print(f"[{scenario_name.upper()}] Memproses Adaptive Log-Chromaticity...")
    stabilized_images = adaptive_log_chromaticity_reduction(raw_images, window_size=window_size)

    # Save outputs
    for path, img_out in zip(valid_paths, stabilized_images):
        filename = os.path.basename(path)
        cv2.imwrite(os.path.join(output_folder, filename), img_out)

    print(f"[{scenario_name.upper()}] Selesai! Hasil disimpan di: '{output_folder}'\n")


# --- Eksekusi Program ---
base_path = "data_penelitian"
scenarios = ["cctv", "medical", "underwater"] # atau ["indoor", "outdoor", "underwater"]

for scn in scenarios:
    input_dir = os.path.join(base_path, scn, "input")
    output_dir = os.path.join(base_path, scn, "output_logch")
    
    process_dataset(scn, input_dir, output_dir, window_size=5)
