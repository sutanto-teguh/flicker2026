import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def get_feature_matches(img1, img2, detector_type='sift'):
    """
    Mendeteksi keypoints, melakukan pencocokan fitur, dan mengisolasi inliers (RANSAC).
    """
    if detector_type.lower() == 'sift':
        detector = cv2.SIFT_create()
        norm_type = cv2.NORM_L2
    else:
        detector = cv2.ORB_create(nfeatures=3000)
        norm_type = cv2.NORM_HAMMING

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2

    kp1, des1 = detector.detectAndCompute(gray1, None)
    kp2, des2 = detector.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return None, 0, 0

    bf = cv2.BFMatcher(norm_type)
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    valid_matches = 0
    inlier_mask = None

    if len(good_matches) >= 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if mask is not None:
            inlier_mask = mask.ravel().tolist()
            valid_matches = int(np.sum(inlier_mask))

    draw_params = dict(
        matchColor=(0, 255, 0),       # Garis hijau untuk inliers valid
        singlePointColor=(255, 0, 0),  # Titik merah untuk keypoints
        matchesMask=inlier_mask,
        flags=cv2.DrawMatchesFlags_DEFAULT
    )
    
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, **draw_params)
    img_matches_rgb = cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB)
    
    return img_matches_rgb, len(kp1) + len(kp2), valid_matches


def create_feature_comparison_grid(scenario_name, base_path, detector_type='sift', save_filename="fitur_komparasi.png"):
    """
    Membuat grid komparasi 2x2 membandingkan hasil pencocokan fitur 4 metode.
    """
    methods = {
        "Raw (Input)": "input",
        "Global HE": "output_ghe",
        "CLAHE": "output_clahe",
        "Proposed Log-Ch": "output_logch"
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), dpi=300)
    axes = axes.ravel()

    for idx, (title, folder_name) in enumerate(methods.items()):
        folder_path = os.path.join(base_path, scenario_name, folder_name)
        
        # Cari file gambar
        exts = ('*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG')
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(folder_path, ext)))
        files = sorted(list(set(files)))

        if len(files) < 2:
            axes[idx].text(0.5, 0.5, f"Data Tidak Ditemukan\n({folder_name})", 
                           ha='center', va='center', fontsize=12)
            axes[idx].axis('off')
            continue

        img1 = cv2.imread(files[0])
        img2 = cv2.imread(files[1])

        match_vis, total_kp, valid_matches = get_feature_matches(img1, img2, detector_type=detector_type)

        if match_vis is not None:
            axes[idx].imshow(match_vis)
            axes[idx].set_title(f"({chr(97+idx)}) {title}\nTotal Keypoints: {total_kp} | Valid Matches (Inliers): {valid_matches}", 
                                fontsize=10, fontweight='bold', pad=8)
        else:
            axes[idx].text(0.5, 0.5, f"Gagal Deteksi Fitur\n({title})", ha='center', va='center')

        axes[idx].axis('off')

    plt.suptitle(f"Perbandingan Visualisasi Pencocokan Fitur ({detector_type.upper()}) - Skenario {scenario_name.upper()}", 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Simpan Gambar Resolusi Tinggi
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    print(f"Visualisasi berhasil disimpan: {save_filename}")
    plt.show()


# --- EKSEKUSI PROGRAM ---
base_dir = "data_penelitian"
skenario = "underwater"  # Ubah ke: "cctv", "medic", atau "underwater"

create_feature_comparison_grid(
    scenario_name=skenario, 
    base_path=base_dir, 
    detector_type='sift', 
    save_filename=f"fig_feature_matching_{skenario}.png"
)