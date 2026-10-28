# Image Processing Tools: RGB to Grayscale & BMP to JPG Converter
By. Teguh S - Universitas Dinamika
Kumpulan utilitas pengolahan citra digital berbasis desktop GUI dengan Python. Repository ini mencakup dua alat utama:
1. **RGB to Grayscale Converter**: Mengonversi citra warna menjadi monokrom menggunakan berbagai pendekatan matematis citra digital.
2. **BMP to JPG Converter & Compressor**: Mengonversi citra format bitmap (BMP) ke JPEG dengan fitur optimasi ukuran file dan kompresi cerdas.

Keduanya dirancang dengan antarmuka grafis yang bersih (*Tkinter/ttk*), pemrosesan asinkron (*multithreading*), pemrosesan massal (*batch processing*), serta kontrol resolusi dan kompresi yang fleksibel.

---

## Daftar Isi
- [Fitur Utama](#fitur-utama)
- [Landasan Teori Grayscale](#landasan-teori-grayscale)
- [Struktur Proyek](#struktur-proyek)
- [Prasyarat & Dependensi](#prasyarat--dependensi)
- [Instalasi](#instalasi)
- [Panduan Penggunaan](#panduan-penggunaan)
  - [RGB to Grayscale Converter](#1-rgb-to-grayscale-converter)
  - [BMP to JPG Converter](#2-bmp-to-jpg-converter)
- [Penjelasan Kompresi Target Ukuran File](#penjelasan-kompresi-target-ukuran-file)
- [Lisensi](#lisensi)

---

## Fitur Utama

- **GUI Responsif & Modern**: Antarmuka desktop menggunakan Tkinter dan tema `ttk` yang ringan tanpa ketergantungan library GUI pihak ketiga yang berat.
- **Background Multithreading**: Pemrosesan citra berjalan di latar belakang (*daemon thread*) sehingga antarmuka tetap responsif tanpa freeze atau *not responding*.
- **Pemrosesan Batch**: Mendukung pemrosesan satu berkas, banyak berkas sekaligus (*multi-selection*), atau pemindaian seluruh isi direktori/folder.
- **Ragam Metode Konversi Grayscale**:
  - Standar NTSC/PAL/BT.601
  - Standar HDTV/BT.709
  - Average (Rata-rata aritmatika)
  - Desaturasi / Lightness
  - Ekstraksi Kanal Mandiri (Kanal R, G, atau B)
- **Kontrol Dimensi & Resolusi**:
  - Skala persentase ($1\% - 200\%$)
  - Pembatasan dimensi maksimum (Max Width $\times$ Max Height) dengan preservasi rasio aspek (*aspect ratio preservation*).
- **Kompresi Pintar (Target File Size in KB)**: Menggunakan algoritma pencarian biner (*binary search*) untuk mencari parameter *quality factor* JPEG terbaik agar ukuran berkas mendekati target kapasitas penyimpanan yang ditentukan pengguna.
- **Logging Real-Time**: Dilengkapi konsol status dan *progress bar* untuk memantau kemajuan konversi setiap berkas.

---

## Landasan Teori Grayscale

Pengonversian citra warna RGB (3 kanal: Red, Green, Blue) menjadi citra keabuan (1 kanal: Luminansi/Intensitas) dapat dilakukan melalui beberapa formulasi matematis:

### 1. Luminansi Standar (ITU-R BT.601)
Metode ini memperhitungkan sensitivitas fisiologis mata manusia yang paling peka terhadap warna hijau, moderat terhadap merah, dan paling rendah terhadap biru:
$$Y = 0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B$$

### 2. Luminansi HDTV / sRGB (ITU-R BT.709)
Metode standar modern yang diterapkan pada panel display digital dan siaran definisi tinggi:
$$Y = 0.2126 \cdot R + 0.7152 \cdot G + 0.0722 \cdot B$$

### 3. Rata-Rata Aritmatika (Average)
Menghitung rata-rata nilai ketiga komponen warna secara seragam:
$$Y = \frac{R + G + B}{3}$$

### 4. Desaturasi (Lightness / HSL Midpoint)
Menghitung titik tengah antara nilai intensitas warna tertinggi dan terendah pada tiap piksel:
$$Y = \frac{\max(R, G, B) + \min(R, G, B)}{2}$$

### 5. Ekstraksi Kanal Tunggal
Mengisolasi salah satu komponen saluran warna untuk dijadikan intensitas monokrom murni:
- Kanal Merah: $Y = R$
- Kanal Hijau: $Y = G$
- Kanal Biru: $Y = B$

---

## Struktur Proyek

```text
.
├── rgb_to_grayscale_converter.py  # Skrip aplikasi konverter RGB ke Grayscale
├── bmp_to_jpg_converter.py        # Skrip aplikasi konverter BMP ke JPG
├── requirements.txt               # Daftar dependensi Python
└── README.md                      # Dokumentasi proyek
```

---

## Prasyarat & Dependensi

Program ini membutuhkan **Python 3.8** atau versi yang lebih baru.

Pustaka yang digunakan:
- `Pillow` (Pillow $\ge$ 9.0.0): Pemrosesan dan manipulasi citra.
- `numpy` ($\ge$ 1.20.0): Komputasi matriks citra cepat untuk algoritma grayscale kustom.
- `tkinter`: Pustaka antarmuka GUI standar (sudah terintegrasi pada instalasi resmi Python).

---

## Instalasi

1. **Clone repository ini ke komputer lokal Anda:**
   ```bash
   git clone https://github.com/username-anda/image-processing-tools.git
   cd image-processing-tools
   ```

2. **(Opsional) Buat dan aktifkan lingkungan virtual (virtual environment):**
   - **Linux / macOS:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   - **Windows:**
     ```cmd
     python -m venv venv
     venv\Scripts\activate
     ```

3. **Pasang paket dependensi:**
   ```bash
   pip install -r requirements.txt
   ```
   *Atau pasang manual jika tanpa requirements.txt:*
   ```bash
   pip install Pillow numpy
   ```

---

## Panduan Penggunaan

### 1. RGB to Grayscale Converter

Jalankan program melalui terminal/command prompt:
```bash
python rgb_to_grayscale_converter.py
```

**Langkah Pengoperasian:**
1. Klik **"Pilih Gambar..."** untuk memilih satu atau beberapa citra, atau **"Pilih Seluruh Folder..."** untuk memproses satu direktori.
2. Tentukan **Metode Konversi** pada menu dropdown (contoh: *Luminance BT.601*, *BT.709*, *Average*, atau *Channel Extraction*).
3. Pilih **Format Output** yang diinginkan (`JPG`, `PNG`, atau `BMP`).
4. Sesuaikan ukuran dan optimasi:
   - Pilih *Kualitas Manual* (slider 10-100) atau centang *Batasi Ukuran Target (KB)* (khusus format JPG).
   - Tentukan ukuran citra (Ukuran Asli, Skala %, atau Batas Maksimum WxH).
5. Tentukan folder penyimpanan hasil atau centang opsi simpan di folder yang sama (nama berkas otomatis ditambahkan imbuhan `_gray`).
6. Klik **"Mulai Konversi ke Grayscale"** dan pantau bilah progres.

---

### 2. BMP to JPG Converter

Jalankan program melalui terminal/command prompt:
```bash
python bmp_to_jpg_converter.py
```

**Langkah Pengoperasian:**
1. Klik **"Pilih File BMP..."** atau **"Pilih Seluruh Folder..."**.
2. Konfigurasikan rasio kompresi:
   - **Kualitas Manual**: Memilih tingkat kompresi statis (slider kualitas 10–100).
   - **Batasi Ukuran Target**: Menentukan ukuran berkas output maksimum (misal: `300` KB).
3. Pilih opsi pengubahan dimensi bila diperlukan.
4. Tentukan folder tujuan penyimpanan.
5. Klik **"Mulai Konversi ke JPG"**.

---

## Penjelasan Kompresi Target Ukuran File

Fitur pembatasan ukuran file target bekerja melalui proses **Optimasi Biner Berulang (Binary Search Compression)**:
1. Program menguji kompresi file sementara di dalam memori (*buffer RAM / `io.BytesIO`*) pada rentang kualitas $Q \in [10, 95]$.
2. Ukuran byte hasil uji dibandingkan terhadap nilai target:
   - Jika ukuran $\le$ target, program mencoba meningkatkan kualitas ke nilai tengah yang lebih tinggi.
   - Jika ukuran $>$ target, program menurunkan rentang kualitas.
3. Setelah 7 siklus iterasi pencarian, program menemukan faktor kualitas tertinggi yang menghasilkan ukuran file di bawah atau paling mendekati batas target pengguna tanpa perlu menulis file uji berulang kali ke hard drive/SSD.

---

## Lisensi

Proyek ini didistribusikan di bawah lisensi **MIT License**. Silakan gunakan, modifikasi, dan kembangkan untuk kebutuhan akademis, komersial, maupun pribadi.
