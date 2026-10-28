import cv2
import numpy as np
import os
import glob

def extract_and_match_features(img1, img2, method='sift'):
    """
    Ekstraksi fitur dan pencocokan antara dua bingkai berurutan.
    
    Parameters:
        img1, img2: Citra input (BGR atau Grayscale).
        method: 'sift' atau 'orb'.
    """
    # 1. Inisialisasi Detektor Fitur
    if method.lower() == 'sift':
        detector = cv2.SIFT_create()
        norm_type = cv2.NORM_L2
    elif method.lower() == 'orb':
        detector = cv2.ORB_create(nfeatures=3000)
        norm_type = cv2.NORM_HAMMING
    elif method.lower() == 'surf':
        try:
            detector = cv2.xfeatures2d.SURF_create(400)
            norm_type = cv2.NORM_L2
        except AttributeError:
            print("⚠️ SURF tidak tersedia di versi OpenCV ini. Menggunakan SIFT...")
            detector = cv2.SIFT_create()
            norm_type = cv2.NORM_L2
    else:
        raise ValueError("Metode harus 'sift', 'surf', atau 'orb'")

    # Konversi ke Grayscale untuk ekstraksi titik kunci
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2

    # 2. Deteksi Keypoints & Deskriptor
    kp1, des1 = detector.detectAndCompute(gray1, None)
    kp2, des2 = detector.detectAndCompute(gray2, None)

    total_keypoints = len(kp1) + len(kp2)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return 0, 0, 0.0, None

    # 3. Pencocokan Fitur dengan BFMatcher + Lowe's Ratio Test
    bf = cv2.BFMatcher(norm_type)
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    ratio_thresh = 0.75  # Lowe's ratio
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)

    # 4. Eliminasi Outlier Menggunakan RANSAC (Mencari Valid Matches)
    valid_matches = 0
    inlier_mask = None
    
    if len(good_matches) >= 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Estimasi Homografi dengan RANSAC
        _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if mask is not None:
            inlier_mask = mask.ravel().tolist()
            valid_matches = int(np.sum(inlier_mask))

    # 5. Hitung Precision Ratio (%)
    total_good = len(good_matches)
    precision_ratio = (valid_matches / total_good * 100.0) if total_good > 0 else 0.0

    # 6. Visualisasi Hasil (Inliers ditandai warna hijau)
    draw_params = dict(
        matchColor=(0, 255, 0),       # Garis hijau untuk valid matches (inliers)
        singlePointColor=(255, 0, 0),  # Titik merah untuk keypoints
        matchesMask=inlier_mask,       # Hanya gambar inliers
        flags=cv2.DrawMatchesFlags_DEFAULT
    )
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, **draw_params)

    return total_keypoints, valid_matches, precision_ratio, img_matches


def evaluate_dataset_features(scenario_name, input_folder, output_vis_folder, detector_type='sift'):
    if not os.path.exists(output_vis_folder):
        os.makedirs(output_vis_folder)

    extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff',
                  '*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.TIF', '*.TIFF')
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(input_folder, ext)))
    image_paths = sorted(list(set(image_paths)))

    if len(image_paths) < 2:
        print(f"[{scenario_name.upper()}] Peringatan: Gambar < 2 pada '{input_folder}'")
        return

    print(f"\n--- Memproses Fitur ({detector_type.upper()}) Skenario: {scenario_name.upper()} ---")
    
    total_kp_list = []
    valid_matches_list = []
    precision_list = []

    for i in range(len(image_paths) - 1):
        path1, path2 = image_paths[i], image_paths[i+1]
        img1 = cv2.imread(path1)
        img2 = cv2.imread(path2)

        if img1 is None or img2 is None:
            continue

        tot_kp, val_match, prec_ratio, vis_img = extract_and_match_features(img1, img2, method=detector_type)

        total_kp_list.append(tot_kp)
        valid_matches_list.append(val_match)
        precision_list.append(prec_ratio)

        # Simpan sampel visualisasi pasangan pertama
        if i == 0 and vis_img is not None:
            save_name = f"match_pair_0_{scenario_name}_{detector_type}.png"
            cv2.imwrite(os.path.join(output_vis_folder, save_name), vis_img)

    avg_kp = np.mean(total_kp_list) if total_kp_list else 0
    avg_valid = np.mean(valid_matches_list) if valid_matches_list else 0
    avg_prec = np.mean(precision_list) if precision_list else 0

    print(f"Rata-rata Total Keypoints : {avg_kp:.1f}")
    print(f"Rata-rata Valid Matches   : {avg_valid:.1f}")
    print(f"Rata-rata Precision Ratio : {avg_prec:.2f}%")


# --- Eksekusi Program ---
base_path = "data_penelitian"
scenarios = ["cctv", "medical", "underwater"]
DETECTOR = "sift"  # Opsional: 'sift' atau 'orb'

for scn in scenarios:
    # Membaca data hasil Log-Ch
    input_dir = os.path.join(base_path, scn, "output_logch")
    vis_dir = os.path.join(base_path, scn, "feature_vis")
    
    if os.path.exists(input_dir):
        evaluate_dataset_features(scn, input_dir, vis_dir, detector_type=DETECTOR)
    else:
        print(f"Folder tidak ditemukan: {input_dir}")