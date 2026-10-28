#!/usr/bin/env python3
"""
BMP to JPG Converter
--------------------
Program Python untuk mengonversi format citra dari BMP ke JPG/JPEG.
Mendukung:
1. Konversi file tunggal.
2. Konversi massal (batch processing) untuk semua file BMP dalam satu folder.
3. Penanganan mode warna (RGBA/P/Grayscale ke RGB standar JPEG).
4. Kustomisasi kualitas kompresi JPEG.

Dependensi:
    pip install Pillow
"""

import os
import sys
from pathlib import Path
from PIL import Image


def convert_bmp_to_jpg(input_path: str, output_path: str = None, quality: int = 95) -> bool:
    """
    Mengonversi satu file citra BMP ke format JPG.

    Args:
        input_path (str): Path file citra BMP sumber.
        output_path (str, optional): Path tujuan file JPG. Jika None, nama file sama dengan ekstensi .jpg.
        quality (int): Kualitas kompresi JPG (1-100, default 95).

    Returns:
        bool: True jika berhasil, False jika gagal.
    """
    try:
        input_file = Path(input_path)
        if not input_file.exists():
            print(f"[ERROR] File '{input_path}' tidak ditemukan.")
            return False

        # Tentukan output path jika tidak diberikan
        if output_path is None:
            output_file = input_file.with_suffix(".jpg")
        else:
            output_file = Path(output_path)

        # Buka gambar menggunakan Pillow
        with Image.open(input_file) as img:
            # Format JPEG tidak mendukung transparansi (Alpha channel)
            # Konversi RGBA, LA, atau P ke RGB dengan background putih jika ada transparansi
            if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                # Buat background putih
                background = Image.new("RGB", img.size, (255, 255, 255))
                # Paste gambar asli di atas background putih menggunakan alpha mask
                alpha_channel = img.convert("RGBA").split()[-1]
                background.paste(img.convert("RGB"), mask=alpha_channel)
                rgb_img = background
            elif img.mode != "RGB":
                # Konversi mode grayscale (L) atau palet biasa ke RGB
                rgb_img = img.convert("RGB")
            else:
                rgb_img = img

            # Simpan citra dalam format JPEG
            rgb_img.save(output_file, format="JPEG", quality=quality, optimize=True)

        print(f"[SUKSES] Berhasil dikonversi: '{input_file.name}' -> '{output_file.name}'")
        return True

    except Exception as e:
        print(f"[GAGAL] Gagal mengonversi '{input_path}': {e}")
        return False


def batch_convert_bmp_to_jpg(folder_path: str, quality: int = 95, recursive: bool = False) -> None:
    """
    Mengonversi semua file BMP yang ada di dalam sebuah direktori menjadi JPG.

    Args:
        folder_path (str): Direktori target yang berisi file BMP.
        quality (int): Kualitas kompresi JPG (1-100).
        recursive (bool): Jika True, mencari file BMP hingga ke subfolder.
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        print(f"[ERROR] Direktori '{folder_path}' tidak valid.")
        return

    # Pola pencarian file .bmp (case-insensitive)
    pattern = "**/*.[bB][mM][pP]" if recursive else "*.[bB][mM][pP]"
    bmp_files = list(folder.glob(pattern))

    if not bmp_files:
        print(f"[INFO] Tidak ditemukan file BMP di '{folder_path}'.")
        return

    print(f"\n[INFO] Ditemukan {len(bmp_files)} file BMP. Memulai konversi...")
    success_count = 0

    for file_path in bmp_files:
        if convert_bmp_to_jpg(str(file_path), quality=quality):
            success_count += 1

    print(f"\n[SELESAI] Konversi selesai: {success_count}/{len(bmp_files)} file berhasil dikonversi.\n")


def main():
    """Menu antarmuka interaktif sederhana."""
    print("=" * 55)
    print("        PROGRAM KONVERSI CITRA BMP KE JPG")
    print("=" * 55)
    print("1. Konversi 1 File BMP")
    print("2. Konversi Semua BMP dalam Folder (Batch)")
    print("3. Keluar")
    print("=" * 55)

    pilihan = input("Pilih mode (1/2/3): ").strip()

    if pilihan == "1":
        file_path = input("Masukkan path file BMP: ").strip().strip('"')
        kualitas = input("Kualitas JPG (1-100, tekan Enter untuk default 95): ").strip()
        quality_val = int(kualitas) if kualitas.isdigit() and 1 <= int(kualitas) <= 100 else 95
        convert_bmp_to_jpg(file_path, quality=quality_val)

    elif pilihan == "2":
        folder_path = input("Masukkan path folder: ").strip().strip('"')
        rekursif = input("Sertakan sub-folder? (y/n, default: n): ").strip().lower() == "y"
        kualitas = input("Kualitas JPG (1-100, tekan Enter untuk default 95): ").strip()
        quality_val = int(kualitas) if kualitas.isdigit() and 1 <= int(kualitas) <= 100 else 95
        batch_convert_bmp_to_jpg(folder_path, quality=quality_val, recursive=rekursif)

    elif pilihan == "3":
        print("Program dihentikan.")
        sys.exit(0)
    else:
        print("Pilihan tidak valid.")


if __name__ == "__main__":
    main()