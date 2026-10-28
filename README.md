# MultiDim-FlickerGuard

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![License: DJKI Copyright](https://img.shields.io/badge/Hak%20Cipta-DJKI%20Kemenkumham-orange.svg)](#hak-cipta-dan-sitasi)

**MultiDim-FlickerGuard** adalah perangkat lunak pemrosesan citra digital (*digital image processing*) dan visi komputer berbasis Python yang dirancang untuk menstabilkan fluktuasi luminans dinamis (*illumination flicker removal*) serta mengoptimalkan akurasi dan presisi registrasi citra sekuensial ($t \rightarrow t+1$). 

Perangkat lunak ini dioptimalkan khusus untuk mengatasi gangguan pencahayaan ekstrem pada 3 domain terapan utama:
1. **Surveilans (CCTV):** Fluktuasi lampu artifisial AC, tutupan awan cepat, dan penyesuaian sensitivitas otomatis sensor (*auto-exposure*).
2. **Pencitraan Medis:** Pantulan basah (*specular highlight/glare*) pada fluoroskopi/endoskopi dan fluktuasi dosis radiasi sensor.
3. **Bawah Air (Underwater):** Refraksi sinar matahari dinamis akibat riak permukaan air laut (*solar caustics*).

---

## Daftar Isi
- [Latar Belakang Masalah](#latar-belakang-masalah)
- [Fitur Utama](#fitur-utama)
- [Arsitektur Sistem](#arsitektur-sistem)
- [Struktur Direktori Repositori](#struktur-direktori-repositori)
- [Instalasi & Prasyarat](#instalasi--prasyarat)
- [Panduan Penggunaan](#panduan-penggunaan)
  - [1. Menjalankan Dashboard GUI](#1-menjalankan-dashboard-gui)
  - [2. Struktur Folder Dataset](#2-struktur-folder-dataset)
  - [3. Langkah Operasional](#3-langkah-operasional)
- [Metrik Evaluasi](#metrik-evaluasi)
- [Hak Cipta dan Sitasi](#hak-cipta-dan-sitasi)

---

## Latar Belakang Masalah

Fluktuasi intensitas luminans antar-bingkai berurutan menghasilkan gradien semu pada bidang citra. Akibatnya:
- Algoritma deteksi fitur standar (SIFT, ORB, SURF) mengekstrak titik-titik kunci palsu (*false keypoints*).
- Jumlah pasangan titik valid (*inliers*) merosot drastis dan rasio presisi kecocokan menurun.
- Kesalahan estimasi geometris temporal (*Normalized Intensity RMSE*) meningkat tajam.

Metode konvensional seperti **Global Histogram Equalization (GHE)** menimbulkan pergeseran rona warna (*color shifting*) dan amplifikasi derau (*noise*), sedangkan **CLAHE** langsung pada domain RGB mendegradasi warna alami citra. 

**MultiDim-FlickerGuard** menyelesaikan permasalahan tersebut dengan memisahkan komponen luminans secara terisolasi pada ruang warna ortogonal (**LAB** / **YCrCb**) serta menerapkan model **Proposed Adaptive Log-Chrominance (Log-Ch)** yang menggabungkan dekomposisi logaritmik, filter spasial-temporal 3D, dan penyelarasan statistik kontras (*contrast matching*).

---

## Fitur Utama

- **Pemisahan Intensitas-Krominans Murni:** Pemrosesan flicker difokuskan secara khusus pada kanal luminans ($L$ atau $Y$) dengan mengunci kanal krominans ($a,b$ atau $Cr,Cb$) untuk menjamin ketiadaan distorsi warna (*zero color shifting*).
- **Model Adaptif Log-Chrominance:** Pemodelan aditif iluminasi-reflektansi dalam domain logaritmik $\ln(I + 1.0)$, estimasi sebaran iluminasi 2D via Gaussian Blur, serta peredaman fluktuasi temporal menggunakan moving average 3D.
- **Dukungan Format Multi-Kanal Otomatis:** Deteksi kedalaman kanal dinamis untuk citra 1-kanal (Grayscale medis), 3-kanal (BGR standar), hingga konversi otomatis 4-kanal (RGBA/BGRA ke BGR).
- **Registrasi Fitur Kokoh:** Ekstraksi titik kunci invarian SIFT, penyaringan pasangan via *Lowe's Distance Ratio Test* ($0.75$), dan estimasi matriks homografi via penapis RANSAC (*error threshold* $5.0$ piksel).
- **Master Controller & GUI Terintegrasi:** Dilengkapi antarmuka grafis Tkinter yang mengotomatisasi eksekusi seluruh metode (*Raw, GHE, CLAHE, Proposed Log-Ch*), menampilkan kurva grafik tren temporal RMSE, tabel metrik, serta ekspor laporan CSV.

---

## Arsitektur Sistem

Pipeline sistem terbagi ke dalam empat lapisan utama (*layered pipeline architecture*):

```
[Sekuens Citra Masukan]
          │
          ▼
┌────────────────────────────────────────────────────────┐
│ 1. DATA INGESTION LAYER                                │
│    - Pembacaan berkas dinamis (.png, .jpg, .tif, dll.) │
│    - Standardisasi 4-kanal RGBA -> 3-kanal BGR         │
│    - Isolasi langsung citra 1-kanal Grayscale medis    │
└────────────────────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────────────────────┐
│ 2. PRE-PROCESSING & COLOR SPACE LAYER                  │
│    - Konversi Ruang Warna: LAB (Kanal L) / YCrCb (Y)   │
│    - Ekualisasi Luminans: GHE, CLAHE, Proposed Log-Ch  │
│    - Rekonstruksi Gamut (Penguncian Krominans Asli)    │
└────────────────────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────────────────────┐
│ 3. FEATURE MATCHING & REGISTRATION LAYER               │
│    - Ekstraksi Fitur Invarian (SIFT)                   │
│    - FLANN Matching + Lowe's Ratio Test (threshold 0.75│
│    - Penapisan Outliers & Estimasi Homografi via RANSAC│
└────────────────────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────────────────────┐
│ 4. EVALUATION & OUTPUT LAYER                           │
│    - Kuantifikasi Temporal RMSE & Normalized RMSE      │
│    - Total Keypoints, Valid Inliers & Precision Ratio  │
│    - Visualisasi Matriks Grid 2x2 & Grafik Tren RMSE   │
└────────────────────────────────────────────────────────┘
```

---

## Struktur Direktori Repositori

```text
MultiDim-FlickerGuard/
├── main_flickerguard_gui.py       # Master Controller GUI Utama
├── data_penelitian/               # Direktori Basis Dataset
│   ├── cctv/
│   │   ├── input/                 # Citra masukan sekuens CCTV
│   │   ├── output_ghe/            # Hasil stabilisasi Global HE
│   │   ├── output_clahe/          # Hasil stabilisasi CLAHE
│   │   └── output_logch/          # Hasil Proposed Log-Chrominance
│   ├── medical/
│   │   ├── input/                 # Citra masukan diagnostik medis
│   │   ├── output_ghe/
│   │   ├── output_clahe/
│   │   └── output_logch/
│   └── underwater/
│       ├── input/                 # Citra masukan bawah air
│       ├── output_ghe/
│       ├── output_clahe/
│       └── output_logch/
├── modules/                       # Modul skrip penelitian individual
│   ├── logch.py                   # Algoritma Proposed Adaptive Log-Ch
│   ├── clahe.py                   # Algoritma CLAHE pada kanal L (LAB)
│   ├── ghe.py                     # Algoritma GHE pada kanal Y (YCrCb)
│   ├── ransac.py                  # Pencocokan fitur & penapisan RANSAC
│   ├── rsme.py                    # Perhitungan Normalized Temporal RMSE
│   ├── grafik_rsme.py             # Generator grafik tren RMSE IEEE style
│   └── metrik_registrasi.py       # Rekapitulasi metrik inliers dan presisi
├── requirements.txt               # Daftar dependensi pustaka Python
└── README.md                      # Dokumentasi teknis proyek
```

---

## Instalasi & Prasyarat

Pastikan sistem Anda telah terpasang **Python 3.8** atau versi yang lebih baru.

1. **Klon Repositori:**
   ```bash
   git clone https://github.com/username/MultiDim-FlickerGuard.git
   cd MultiDim-FlickerGuard
   ```

2. **Buat dan Aktifkan Virtual Environment (Direkomendasikan):**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Pasang Dependensi:**
   ```bash
   pip install -r requirements.txt
   ```

   *Isi berkas `requirements.txt`:*
   ```text
   opencv-python>=4.8.0
   numpy>=1.24.0
   matplotlib>=3.7.0
   pandas>=2.0.0
   ```

---

## Panduan Penggunaan

### 1. Menjalankan Dashboard GUI
Jalankan master controller melalui terminal/command prompt:
```bash
python main_flickerguard_gui.py
```

### 2. Struktur Folder Dataset
Tempatkan sekuens citra yang akan diuji ke dalam subdirektori masukan domain terkait:
- **CCTV:** `data_penelitian/cctv/input/`
- **Medis:** `data_penelitian/medical/input/`
- **Underwater:** `data_penelitian/underwater/input/`

Format berkas yang didukung mencakup: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, dan `.tiff`.

### 3. Langkah Operasional
1. **Basis Direktori:** Klik tombol **Pilih...** dan arahkan ke folder `data_penelitian` (sistem GUI juga mendukung deteksi pintar apabila Anda langsung mengarahkan ke folder `input`).
2. **Skenario Domain:** Pilih skenario pada dropdown (`cctv`, `medical`, atau `underwater`).
3. **Load Sekuens Citra:** Klik tombol untuk memindai berkas dan mengonversi kedalaman kanal.
4. **Jalankan Semua Pipeline:** Sistem akan menjalankan pemrosesan *GHE*, *CLAHE*, dan *Proposed Log-Ch* di latar belakang (*background thread*), menghitung homografi RANSAC, serta mengukur stabilitas temporal RMSE.
5. **Eksplorasi Hasil:**
   - **Tab 1 (Tabel Ringkasan Metrik):** Menampilkan tabel perbandingan metrik numerik dan tombol ekspor CSV.
   - **Tab 2 (Grafik Tren Temporal RMSE):** Menampilkan kurva tren penurunan RMSE temporal per urutan transisi bingkai.
   - **Tab 3 (Visualisasi Grid Pencocokan Fitur):** Menampilkan kanvas visualisasi 4 kuadran perbandingan garis kecocokan inliers.

---

## Metrik Evaluasi

Performa stabilisasi dan registrasi dievaluasi berdasarkan dua metrik utama:

1. **Normalized Intensity RMSE (Temporal Stability):**
   $$\text{RMSE}_{\text{norm}} = \frac{1}{255} \sqrt{\frac{1}{H \times W} \sum_{x=1}^{H}\sum_{y=1}^{W} \left(I_t(x, y) - I_{t+1}(x, y)\right)^2}$$
   *Nilai yang semakin mendekati 0 membuktikan fluktuasi kedipan cahaya antar-frame berhasil diredam secara temporal.*

2. **Precision Ratio (Registration Quality):**
   $$\text{Precision Ratio} = \frac{\text{Jumlah Inliers Valid (RANSAC)}}{\text{Total Pasangan Titik Kecocokan (Good Matches)}} \times 100\%$$
   *Persentase presisi yang lebih tinggi menunjukkan stabilitas vektor deskriptor terhadap perubahan iluminasi dinamis.*

---

## Hak Cipta dan Sitasi

Perangkat lunak ini terdaftar secara resmi sebagai Ciptaan Program Komputer pada Direktorat Jenderal Kekayaan Intelektual (DJKI) Kementerian Hukum Republik Indonesia:

- **Judul Ciptaan:** MultiDim-FlickerGuard: Perangkat Lunak Penstabil Fluktuasi Luminans dan Pengoptimal Registrasi Citra untuk Skenario CCTV, Medis, dan Underwater
- **Pencipta:** Teguh Sutanto, Anjik Sukmaaji, Harianto
- **Tahun Ciptaan:** 2026

Jika Anda menggunakan program atau algoritma ini untuk publikasi ilmiah, mohon cantumkan sitasi dokumen penelitian/HKI terkait.
