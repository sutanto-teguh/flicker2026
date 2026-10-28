#!/usr/bin/env python3
"""
BMP to JPG Converter with GUI
-----------------------------
Program Python dengan antarmuka grafis (Tkinter) untuk mengonversi gambar BMP ke JPG/JPEG.
Fitur Unggulan:
1. Konversi file tunggal, multi-file, atau seluruh folder (Batch).
2. Kompresi target ukuran file (contoh: <= 500 KB) dengan algoritma bisection/iterasi kualitas.
3. Pengubahan resolusi/ukuran gambar (Persentase skala atau Dimensi Max WxH dengan aspect ratio).
4. Penanganan mode warna otomatis (RGBA / Indexed / Grayscale ke RGB).
5. Progress bar, status real-time, dan log riwayat konversi.

Dependensi:
    pip install Pillow
"""

import os
import io
import sys
import threading
from pathlib import Path
from PIL import Image

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText


def prepare_rgb_image(img: Image.Image) -> Image.Image:
    """
    Menyiapkan gambar agar kompatibel dengan format JPEG (RGB).
    Menghilangkan transparansi dengan memberi latar belakang putih.
    """
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba_img = img.convert("RGBA")
        background = Image.new("RGB", rgba_img.size, (255, 255, 255))
        alpha = rgba_img.split()[-1]
        background.paste(rgba_img.convert("RGB"), mask=alpha)
        return background
    elif img.mode != "RGB":
        return img.convert("RGB")
    return img.copy()


def resize_image(img: Image.Image, resize_mode: str, scale_pct: float, max_w: int, max_h: int) -> Image.Image:
    """
    Mengubah ukuran gambar berdasarkan konfigurasi pengguna.
    """
    w, h = img.size
    if resize_mode == "none":
        return img

    if resize_mode == "scale":
        factor = max(1.0, scale_pct) / 100.0
        new_w = max(1, int(w * factor))
        new_h = max(1, int(h * factor))
        return img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    if resize_mode == "dimension":
        if max_w <= 0 or max_h <= 0:
            return img
        # Mempertahankan rasio aspek
        ratio = min(max_w / w, max_h / h)
        if ratio < 1.0:  # Hanya perkecil jika melebihi batas
            new_w = max(1, int(w * ratio))
            new_h = max(1, int(h * ratio))
            return img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    return img


def compress_to_target_kb(img: Image.Image, output_path: Path, target_kb: int, fallback_quality: int = 90) -> tuple:
    """
    Mengompresi gambar JPEG agar ukurannya mendekati atau di bawah target_kb (dalam Kilobyte).
    Menggunakan binary search pada rentang kualitas 10 - 95.
    
    Returns:
        tuple: (final_quality, final_size_kb)
    """
    target_bytes = target_kb * 1024
    low_q, high_q = 10, 95
    best_buffer = None
    best_quality = fallback_quality

    for _ in range(7):  # Binary search presisi tinggi dalam 7 iterasi
        mid_q = (low_q + high_q) // 2
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=mid_q, optimize=True)
        size = buf.tell()

        if size <= target_bytes:
            best_buffer = buf.getvalue()
            best_quality = mid_q
            low_q = mid_q + 1  # Coba kualitas lebih tinggi yang masih muat
        else:
            high_q = mid_q - 1

    # Jika bahkan kualitas 10 masih lebih besar dari target, gunakan hasil kompresi terendah
    if best_buffer is None:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=10, optimize=True)
        best_buffer = buf.getvalue()
        best_quality = 10

    with open(output_path, "wb") as f:
        f.write(best_buffer)

    final_size_kb = len(best_buffer) / 1024
    return best_quality, final_size_kb


class BMPConverterGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("BMP to JPG Converter & Compressor")
        self.geometry("780x680")
        self.minsize(680, 560)

        # Styling
        self.style = ttk.Style(self)
        if "clam" in self.style.theme_names():
            self.style.theme_use("clam")

        self.selected_files = []
        self.output_directory = ""
        self.is_processing = False

        self._build_ui()

    def _build_ui(self):
        # Header Frame
        header = ttk.Frame(self, padding=12)
        header.pack(fill=tk.X)
        ttk.Label(
            header,
            text="BMP to JPG Converter & Compressor",
            font=("Helvetica", 16, "bold"),
            foreground="#1e40af"
        ).pack(anchor=tk.W)
        ttk.Label(
            header,
            text="Ubah format citra BMP ke JPG dengan kontrol kualitas, ukuran file target, dan resolusi.",
            font=("Helvetica", 9),
            foreground="#475569"
        ).pack(anchor=tk.W)

        # Main Scrollable / Stacked Container
        main_frame = ttk.Frame(self, padding=(12, 0, 12, 12))
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 1. Source Selection Frame
        src_group = ttk.LabelFrame(main_frame, text=" 1. Pilih Citra Sumber (BMP) ", padding=10)
        src_group.pack(fill=tk.X, pady=4)

        btn_box = ttk.Frame(src_group)
        btn_box.pack(fill=tk.X)
        ttk.Button(btn_box, text="Pilih File BMP...", command=self._select_files).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_box, text="Pilih Seluruh Folder...", command=self._select_folder).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_box, text="Kosongkan", command=self._clear_files).pack(side=tk.RIGHT, padx=4)

        self.lbl_file_count = ttk.Label(src_group, text="Belum ada file yang dipilih.", foreground="#64748b")
        self.lbl_file_count.pack(anchor=tk.W, pady=(6, 0), padx=4)

        # 2. Options Frame (Resize & Compression)
        opt_group = ttk.LabelFrame(main_frame, text=" 2. Pengaturan Kompresi & Ukuran Citra ", padding=10)
        opt_group.pack(fill=tk.X, pady=4)

        # Mode Kompresi
        comp_box = ttk.Frame(opt_group)
        comp_box.pack(fill=tk.X, pady=2)

        self.var_comp_mode = tk.StringVar(value="quality")
        ttk.Radiobutton(
            comp_box, text="Kualitas Manual (1-100):", variable=self.var_comp_mode,
            value="quality", command=self._toggle_compression_inputs
        ).grid(row=0, column=0, sticky=tk.W, padx=4)

        self.var_quality = tk.IntVar(value=90)
        self.slider_quality = ttk.Scale(comp_box, from_=10, to=100, variable=self.var_quality, orient=tk.HORIZONTAL, length=140)
        self.slider_quality.grid(row=0, column=1, padx=4)
        self.lbl_quality_val = ttk.Label(comp_box, textvariable=self.var_quality, width=4)
        self.lbl_quality_val.grid(row=0, column=2, padx=2)

        ttk.Radiobutton(
            comp_box, text="Batasi Ukuran Target (KB):", variable=self.var_comp_mode,
            value="target_kb", command=self._toggle_compression_inputs
        ).grid(row=0, column=3, sticky=tk.W, padx=(20, 4))

        self.var_target_kb = tk.StringVar(value="300")
        self.entry_target_kb = ttk.Entry(comp_box, textvariable=self.var_target_kb, width=8, state=tk.DISABLED)
        self.entry_target_kb.grid(row=0, column=4, padx=4)
        ttk.Label(comp_box, text="KB").grid(row=0, column=5)

        ttk.Separator(opt_group, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        # Mode Resize
        resize_box = ttk.Frame(opt_group)
        resize_box.pack(fill=tk.X, pady=2)

        self.var_resize_mode = tk.StringVar(value="none")
        ttk.Radiobutton(
            resize_box, text="Ukuran Asli", variable=self.var_resize_mode,
            value="none", command=self._toggle_resize_inputs
        ).grid(row=0, column=0, sticky=tk.W, padx=4)

        ttk.Radiobutton(
            resize_box, text="Skala (%):", variable=self.var_resize_mode,
            value="scale", command=self._toggle_resize_inputs
        ).grid(row=0, column=1, sticky=tk.W, padx=(16, 4))

        self.var_scale = tk.StringVar(value="75")
        self.entry_scale = ttk.Entry(resize_box, textvariable=self.var_scale, width=6, state=tk.DISABLED)
        self.entry_scale.grid(row=0, column=2, padx=4)

        ttk.Radiobutton(
            resize_box, text="Max W x H (px):", variable=self.var_resize_mode,
            value="dimension", command=self._toggle_resize_inputs
        ).grid(row=0, column=3, sticky=tk.W, padx=(16, 4))

        self.var_max_w = tk.StringVar(value="1920")
        self.entry_max_w = ttk.Entry(resize_box, textvariable=self.var_max_w, width=6, state=tk.DISABLED)
        self.entry_max_w.grid(row=0, column=4, padx=2)
        ttk.Label(resize_box, text="x").grid(row=0, column=5)
        self.var_max_h = tk.StringVar(value="1080")
        self.entry_max_h = ttk.Entry(resize_box, textvariable=self.var_max_h, width=6, state=tk.DISABLED)
        self.entry_max_h.grid(row=0, column=6, padx=2)

        # 3. Output Destination Frame
        dst_group = ttk.LabelFrame(main_frame, text=" 3. Lokasi Penyimpanan Hasil ", padding=10)
        dst_group.pack(fill=tk.X, pady=4)

        self.var_same_dir = tk.BooleanVar(value=True)
        chk_same = ttk.Checkbutton(
            dst_group, text="Simpan di folder yang sama dengan file asli",
            variable=self.var_same_dir, command=self._toggle_output_dir
        )
        chk_same.pack(anchor=tk.W, pady=2)

        dst_input_box = ttk.Frame(dst_group)
        dst_input_box.pack(fill=tk.X, pady=2)

        self.var_output_path = tk.StringVar()
        self.entry_output = ttk.Entry(dst_input_box, textvariable=self.var_output_path, state=tk.DISABLED)
        self.entry_output.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))

        self.btn_browse_dst = ttk.Button(dst_input_box, text="Pilih Folder...", command=self._select_output_folder, state=tk.DISABLED)
        self.btn_browse_dst.pack(side=tk.RIGHT)

        # 4. Action & Progress
        act_box = ttk.Frame(main_frame, padding=(0, 6))
        act_box.pack(fill=tk.X)

        self.btn_convert = ttk.Button(act_box, text="Mulai Konversi ke JPG", command=self._start_conversion_thread)
        self.btn_convert.pack(fill=tk.X, ipady=5)

        self.prog_bar = ttk.Progressbar(main_frame, orient=tk.HORIZONTAL, mode="determinate")
        self.prog_bar.pack(fill=tk.X, pady=(6, 4))

        self.lbl_status = ttk.Label(main_frame, text="Siap.", foreground="#475569")
        self.lbl_status.pack(anchor=tk.W)

        # 5. Log Console
        log_group = ttk.LabelFrame(main_frame, text=" Log Aktivitas ", padding=6)
        log_group.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

        self.log_text = ScrolledText(log_group, height=6, font=("Consolas", 9), wrap=tk.WORD, bg="#0f172a", fg="#f8fafc")
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _append_log(self, text: str):
        self.log_text.insert(tk.END, text + "\n")
        self.log_text.see(tk.END)

    def _toggle_compression_inputs(self):
        mode = self.var_comp_mode.get()
        if mode == "quality":
            self.slider_quality.config(state=tk.NORMAL)
            self.entry_target_kb.config(state=tk.DISABLED)
        else:
            self.slider_quality.config(state=tk.DISABLED)
            self.entry_target_kb.config(state=tk.NORMAL)

    def _toggle_resize_inputs(self):
        mode = self.var_resize_mode.get()
        if mode == "none":
            self.entry_scale.config(state=tk.DISABLED)
            self.entry_max_w.config(state=tk.DISABLED)
            self.entry_max_h.config(state=tk.DISABLED)
        elif mode == "scale":
            self.entry_scale.config(state=tk.NORMAL)
            self.entry_max_w.config(state=tk.DISABLED)
            self.entry_max_h.config(state=tk.DISABLED)
        elif mode == "dimension":
            self.entry_scale.config(state=tk.DISABLED)
            self.entry_max_w.config(state=tk.NORMAL)
            self.entry_max_h.config(state=tk.NORMAL)

    def _toggle_output_dir(self):
        if self.var_same_dir.get():
            self.entry_output.config(state=tk.DISABLED)
            self.btn_browse_dst.config(state=tk.DISABLED)
        else:
            self.entry_output.config(state=tk.NORMAL)
            self.btn_browse_dst.config(state=tk.NORMAL)

    def _select_files(self):
        files = filedialog.askopenfilenames(
            title="Pilih File BMP",
            filetypes=[("Bitmap Files (*.bmp)", "*.bmp;*.BMP"), ("All Files (*.*)", "*.*")]
        )
        if files:
            self.selected_files = [Path(f) for f in files]
            self.lbl_file_count.config(text=f"{len(self.selected_files)} file terpilih.")
            self._append_log(f"[INFO] Dipilih {len(self.selected_files)} file.")

    def _select_folder(self):
        folder = filedialog.askdirectory(title="Pilih Folder Berisi File BMP")
        if folder:
            dir_path = Path(folder)
            found = list(dir_path.glob("*.[bB][mM][pP]"))
            if not found:
                messagebox.showwarning("Peringatan", "Tidak ditemukan file BMP pada direktori tersebut.")
                return
            self.selected_files = found
            self.lbl_file_count.config(text=f"{len(self.selected_files)} file ditemukan di '{dir_path.name}'.")
            self._append_log(f"[INFO] Ditemukan {len(self.selected_files)} file BMP di folder {folder}.")

    def _select_output_folder(self):
        folder = filedialog.askdirectory(title="Pilih Folder Penyimpanan Output")
        if folder:
            self.output_directory = folder
            self.var_output_path.set(folder)

    def _clear_files(self):
        self.selected_files = []
        self.lbl_file_count.config(text="Belum ada file yang dipilih.")
        self.prog_bar["value"] = 0
        self.lbl_status.config(text="Siap.")

    def _start_conversion_thread(self):
        if not self.selected_files:
            messagebox.showwarning("Peringatan", "Silakan pilih setidaknya satu file BMP terlebih dahulu!")
            return

        if not self.var_same_dir.get() and not self.var_output_path.get().strip():
            messagebox.showwarning("Peringatan", "Silakan tentukan folder tujuan penyimpanan!")
            return

        # Disable tombol proses saat berjalan
        self.btn_convert.config(state=tk.DISABLED)
        self.is_processing = True

        # Jalankan di background thread agar GUI tidak freeze
        thread = threading.Thread(target=self._process_conversion, daemon=True)
        thread.start()

    def _process_conversion(self):
        total_files = len(self.selected_files)
        success_count = 0
        comp_mode = self.var_comp_mode.get()
        target_kb = 300
        manual_quality = self.var_quality.get()

        if comp_mode == "target_kb":
            try:
                target_kb = max(10, int(self.var_target_kb.get()))
            except ValueError:
                target_kb = 300

        resize_mode = self.var_resize_mode.get()
        scale_val = 100.0
        max_w, max_h = 1920, 1080

        if resize_mode == "scale":
            try:
                scale_val = float(self.var_scale.get())
            except ValueError:
                scale_val = 100.0
        elif resize_mode == "dimension":
            try:
                max_w = int(self.var_max_w.get())
                max_h = int(self.var_max_h.get())
            except ValueError:
                max_w, max_h = 1920, 1080

        save_to_same = self.var_same_dir.get()
        custom_out_dir = Path(self.var_output_path.get().strip()) if not save_to_same else None

        self.prog_bar["maximum"] = total_files
        self.prog_bar["value"] = 0

        self._append_log("\n--- Memulai Proses Konversi ---")

        for idx, src_path in enumerate(self.selected_files, start=1):
            try:
                if not src_path.exists():
                    self._append_log(f"[GAGAL] File '{src_path.name}' tidak ditemukan.")
                    continue

                if save_to_same:
                    out_path = src_path.with_suffix(".jpg")
                else:
                    out_path = custom_out_dir / f"{src_path.stem}.jpg"

                # Proses Konversi Gambar
                with Image.open(src_path) as img:
                    orig_size_kb = src_path.stat().st_size / 1024
                    rgb_img = prepare_rgb_image(img)
                    resized_img = resize_image(rgb_img, resize_mode, scale_val, max_w, max_h)

                    if comp_mode == "target_kb":
                        final_q, final_kb = compress_to_target_kb(resized_img, out_path, target_kb)
                        log_msg = f"[OK] {src_path.name} -> {out_path.name} ({orig_size_kb:.1f} KB -> {final_kb:.1f} KB, Q={final_q})"
                    else:
                        resized_img.save(out_path, format="JPEG", quality=manual_quality, optimize=True)
                        final_kb = out_path.stat().st_size / 1024
                        log_msg = f"[OK] {src_path.name} -> {out_path.name} ({orig_size_kb:.1f} KB -> {final_kb:.1f} KB, Q={manual_quality})"

                    self._append_log(log_msg)
                    success_count += 1

            except Exception as e:
                self._append_log(f"[ERROR] Gagal memproses {src_path.name}: {e}")

            # Update Progress Bar & Status
            self.prog_bar["value"] = idx
            self.lbl_status.config(text=f"Memproses {idx}/{total_files} file...")

        self.lbl_status.config(text=f"Selesai: {success_count}/{total_files} file berhasil dikonversi.")
        self._append_log(f"--- Selesai: {success_count}/{total_files} berhasil dikonversi ---\n")
        self.btn_convert.config(state=tk.NORMAL)
        self.is_processing = False

        messagebox.showinfo(
            "Konversi Selesai",
            f"Proses selesai!\n{success_count} dari {total_files} file berhasil dikonversi ke format JPG."
        )


def main():
    app = BMPConverterGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
