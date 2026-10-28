import cv2
import os

def extract_frames(video_path, output_folder, interval_seconds=None):
    """
    Fungsi untuk mengekstrak frame dari video MP4.
    
    :param video_path: Path/lokasi file video MP4
    :param output_folder: Folder tempat menyimpan gambar hasil ekstraksi
    :param interval_seconds: Interval waktu (dalam detik) untuk mengambil gambar.
                             Jika None atau 0, akan mengekstrak SEMUA frame.
    """
    # Buat folder output jika belum ada
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Folder '{output_folder}' berhasil dibuat.")

    # Buka file video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Tidak dapat membuka video '{video_path}'. Pastikan file ada dan formatnya sesuai.")
        return

    # Dapatkan informasi video
    fps = cap.get(cv2.CAP_PROP_FPS) # Frame per detik
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"--- Informasi Video ---")
    print(f"FPS          : {fps:.2f}")
    print(f"Total Frame  : {total_frames}")
    print(f"Durasi       : {duration:.2f} detik")
    print(f"-----------------------")

    # Hitung kelipatan frame yang akan diambil berdasarkan interval
    if interval_seconds and interval_seconds > 0:
        frame_interval = int(fps * interval_seconds)
    else:
        frame_interval = 1  # Ekstrak setiap frame

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()

        # Jika video selesai atau tidak ada frame lagi
        if not ret:
            break

        # Simpan gambar jika sesuai dengan interval
        if frame_count % frame_interval == 0:
            filename = f"frame_{saved_count:04d}.jpg"
            output_filepath = os.path.join(output_folder, filename)
            
            # Simpan frame ke dalam file gambar
            cv2.imwrite(output_filepath, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"\nSelesai! Berhasil menyimpan {saved_count} gambar ke folder '{output_folder}'.")

# --- Penggunaan Program ---
if __name__ == "__main__":
    # Ganti dengan nama/lokasi file video Anda
    PATH_VIDEO = "1784707581765.mp4" 
    
    # Folder tujuan untuk menyimpan hasil gambar
    FOLDER_OUTPUT = "hasil_frame" 
    
    # Contoh 1: Ambil gambar setiap 1 detik
    # Ubah menjadi None jika ingin mengekstrak SEMUA frame tanpa lewati
    INTERVAL_DETIK = 1.0 

    extract_frames(PATH_VIDEO, FOLDER_OUTPUT, interval_seconds=INTERVAL_DETIK)