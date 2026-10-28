import cv2
import numpy as np
import os
import glob

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Menerapkan CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Mendukung citra Grayscale (1-channel) dan Berwarna/BGR (3-channel).
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    # Deteksi citra Grayscale (umum pada data medis seperti X-ray/Fluoroskopi)
    if len(image.shape) == 2 or image.shape[2] == 1:
        return clahe.apply(image)
    
    # Deteksi citra Berwarna / BGR (CCTV dan Underwater)
    # Gunakan ruang warna LAB agar pemrosesan kontras tidak merusak warna
    lab_img = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_img)
    
    # Terapkan CLAHE hanya pada kanal L (Luminance)
    l_clahe = clahe.apply(l)
    
    # Gabungkan kembali kanal dan konversi ke BGR
    updated_lab = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(updated_lab, cv2.COLOR_LAB2BGR)

def process_clahe_reduction(scenario_name, input_folder, output_folder, clip_limit=2.0, tile_grid_size=(8, 8)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff', 
                  '*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.TIF', '*.TIFF')
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, ext)))
    image_files = sorted(list(set(image_files)))

    if not image_files:
        print(f"[{scenario_name.upper()}] Peringatan: Tidak ada gambar di '{input_folder}'")
        return

    print(f"[{scenario_name.upper()}] Memproses {len(image_files)} gambar dengan CLAHE...")
    processed_count = 0

    for img_path in image_files:
        filename = os.path.basename(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            continue
            
        # Penanganan citra RGBA (4 channel) menjadi BGR (3 channel)
        if len(img.shape) == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
        processed_img = apply_clahe(img, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
        cv2.imwrite(os.path.join(output_folder, filename), processed_img)
        processed_count += 1
        
    print(f"[{scenario_name.upper()}] Selesai! {processed_count} gambar disimpan di: '{output_folder}'\n")

# --- Eksekusi Program ---
base_path = "data_penelitian"
scenarios = ["cctv", "medical", "underwater"]

for scn in scenarios:
    input_dir = os.path.join(base_path, scn, "input")
    output_dir = os.path.join(base_path, scn, "output_clahe")
    
    # Parameters: clip_limit=2.0 untuk membatasi amplifikasi noise, tile_grid_size=(8,8) untuk lokalitas
    process_clahe_reduction(scn, input_dir, output_dir, clip_limit=2.0, tile_grid_size=(8, 8))
