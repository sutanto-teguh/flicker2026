#!/usr/bin/env python3
"""
RGB to Grayscale Image Converter with GUI
-----------------------------------------
Program Python dengan antarmuka grafis (Tkinter) untuk mengonversi citra warna (RGB)
menjadi citra derajat keabuan (Grayscale / Monokrom).

Fitur Unggulan:
1. Berbagai Metode Grayscale:
   - Luminance / ITU-R BT.601 (Bobot visual standar: 0.299R + 0.587G + 0.114B)
   - Luminance / ITU-R BT.709 (Standar HDTV: 0.2126R + 0.7152G + 0.0722B)
   - Average / Rata-rata ((R + G + B) / 3)
   - Desaturation / Lightness ((max(R,G,B) + min(R,G,B)) / 2)
   - Kanal Tunggal (Hanya Red, Green, atau Blue)
2. Pilihan Format Output (JPG, PNG, BMP).
3. Pengubahan resolusi (Skala % atau Dimensi Max WxH dengan aspect ratio).
4. Kompresi target ukuran file (KB) untuk format JPEG.
5. Pemrosesan Batch (Banyak file atau seluruh folder) dengan Multithreading.
6. Log riwayat konversi dan progress bar real-time.

Dependensi:
    pip install Pillow numpy
"""

import os
import io
import sys
import threading
from pathlib import Path
from PIL import Image
import numpy as np

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def convert_to_grayscale(img: Image.Image, method: str) -> Image.Image:
    """
    Mengonversi citra input menjadi citra derajat keabuan (mode 'L')
    berdasarkan metode yang dipilih.
    """
    # Pastikan citra dalam format RGB (hilangkan alpha jika ada)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        alpha = rgba.split()[-1]
        bg.paste(rgba.convert("RGB"), mask=alpha)
        rgb_img = bg
    elif img.mode != "RGB":
        rgb_img = img.convert("RGB")
    else:
        rgb_img = img

    if method == "luminance_bt601":
        # Formula resmi ITU-R BT.601 (sama dengan Pillow img.convert('L'))
        return rgb_img.convert("L")

    arr = np.array(rgb_img, dtype=np.float32)
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    if method == "luminance_bt709":
        # Formula HDTV ITU-R BT.709
        gray = 0.2126 * r + 0.7152 * g + 0.0722 * b
    elif method == "average":
        # Rata-rata sederhana
        gray = (r + g + b) / 3.0
    elif method == "desaturation":
        # (max(R, G, B) + min(R, G, B)) / 2
        max_val = np.maximum(np.maximum(r, g), b)
        min_val = np.minimum(np.minimum(r, g), b)
        gray = (max_val + min_val) / 2.0
    elif method == "channel_r":
        gray = r
    elif method == "channel_g":
        gray = g
    elif method == "channel_b":
        gray = b
    else:
        return rgb_img.convert("L")

    gray_uint8 = np.clip(gray, 0, 255).astype(np.uint8)
    return Image.fromarray(gray_uint8, mode="L")


def resize_image(img: Image.Image, resize_mode: str, scale_pct: float, max_w: int, max_h: int) -> Image.Image:
    """
    Mengubah ukuran citra dengan mempertahankan aspek rasio.
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
        ratio = min(max_w / w, max_h / h)
        if ratio < 1.0:
            new_w = max(1, int(w * ratio))
            new_h = max(1, int(h * ratio))
            return img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    return img


def compress_to_target_kb(img: Image.Image, output_path: Path, target_kb: int, fallback_quality: int = 90) -> tuple:
    """
    Melakukan binary search tingkat kualitas JPEG untuk mendekati target ukuran file (KB).
    """
    target_bytes = target_kb * 1024
    low_q, high_q = 10, 95
    best_buffer = None
    best_quality = fallback_quality

    for _ in range(7):
        mid_q = (low_q + high_q) // 2
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=mid_q, optimize=True)
        size = buf.tell()

        if size <= target_bytes:
            best_buffer = buf.getvalue()
            best_quality = mid_q
            low_q = mid_q + 1
        else:
            high_q = mid_q - 1

    if best_buffer is None:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=10, optimize=True)
        best_buffer = buf.getvalue()
        best_quality = 10

    with open(output_path, "wb") as f:
        f.write(best_buffer)

    final_size_kb = len(best_buffer) / 1024
    return best_quality, final_size_kb


class GrayscaleConverterGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RGB to Grayscale Image Converter")
        self.geometry("820x750")
        self.minsize(720, 600)

        self.style = ttk.Style(self)
        if "clam" in self.style.theme_names():
            self.style.theme_use("clam")

        self.selected_files = []
        self.output_directory = ""
        self.is_processing = False

        self._build_ui()

    def _build_ui(self):
        # Header
        header = ttk.Frame(self, padding=12)
        header.pack(fill=tk.X)
        ttk.Label(
            header,
            text="RGB to Grayscale Image Converter",
            font=("Helvetica", 16, "bold"),
            foreground="#0f766e"
        ).pack(anchor=tk.W)
        ttk.Label(
            header,
            text="Ubah gambar warna ke derajat keabuan (Monokrom) dengan berbagai metode ilmiah dan opsi kompresi.",
            font=("Helvetica", 9),
            foreground="#475569"
        ).pack(anchor=tk.W)

        # Main Container
        main_frame = ttk.Frame(self, padding=(12, 0, 12, 12))
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 1. Pilih Citra Sumber
        src_group = ttk.LabelFrame(main_frame, text=" 1. Pilih Gambar Sumber ", padding=10)
        src_group.pack(fill=tk.X, pady=4)

        btn_box = ttk.Frame(src_group)
        btn_box.pack(fill=tk.X)
        ttk.Button(btn_box, text="Pilih Gambar...", command=self._select_files).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_box, text="Pilih Seluruh Folder...", command=self._select_folder).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_box, text="Kosongkan", command=self._clear_files).pack(side=tk.RIGHT, padx=4)

        self.lbl_file_count = ttk.Label(src_group, text="Belum ada gambar yang dipilih.", foreground="#64748b")
        self.lbl_file_count.pack(anchor=tk.W, pady=(6, 0), padx=4)

        # 2. Metode Grayscale & Format Output
        method_group = ttk.LabelFrame(main_frame, text=" 2. Metode Grayscale & Format Output ", padding=10)
        method_group.pack(fill=tk.X, pady=4)

        m_row = ttk.Frame(method_group)
        m_row.pack(fill=tk.X, pady=2)

        ttk.Label(m_row, text="Metode Konversi:").pack(side=tk.LEFT, padx=(4, 8))
        self.var_method = tk.StringVar(value="luminance_bt601")
        method_choices = [
            ("Luminance BT.601 (Standar Persepsi Mata)", "luminance_bt601"),
            ("Luminance BT.709 (Standar HDTV / sRGB)", "luminance_bt709"),
            ("Average / Rata-rata ((R+G+B)/3)", "average"),
            ("Desaturation ((Max+Min)/2)", "desaturation"),
            ("Kanal Merah (Red Channel)", "channel_r"),
            ("Kanal Hijau (Green Channel)", "channel_g"),
            ("Kanal Biru (Blue Channel)", "channel_b"),
        ]
        self.combo_method = ttk.Combobox(
            m_row, textvariable=self.var_method, state="readonly", width=38,
            values=[name for name, _ in method_choices]
        )
        self.combo_method.current(0)
        self.combo_method.pack(side=tk.LEFT, padx=4)
        self._method_map = {name: val for name, val in method_choices}

        ttk.Label(m_row, text="Format Output:").pack(side=tk.LEFT, padx=(20, 8))
        self.var_out_format = tk.StringVar(value="jpg")
        self.combo_fmt = ttk.Combobox(
            m_row, textvariable=self.var_out_format, state="readonly", width=8,
            values=["jpg", "png", "bmp"]
        )
        self.combo_fmt.pack(side=tk.LEFT)
        self.combo_fmt.bind("<<ComboboxSelected>>", lambda e: self._on_format_changed())

        # 3. Kompresi & Dimensi
        opt_group = ttk.LabelFrame(main_frame, text=" 3. Pengaturan Ukuran & Kompresi ", padding=10)
        opt_group.pack(fill=tk.X, pady=4)

        # Kompresi Baris
        comp_box = ttk.Frame(opt_group)
        comp_box.pack(fill=tk.X, pady=2)

        self.var_comp_mode = tk.StringVar(value="quality")
        self.rb_quality = ttk.Radiobutton(
            comp_box, text="Kualitas Manual (1-100):", variable=self.var_comp_mode,
            value="quality", command=self._toggle_compression_inputs
        )
        self.rb_quality.grid(row=0, column=0, sticky=tk.W, padx=4)

        self.var_quality = tk.IntVar(value=90)
        self.slider_quality = ttk.Scale(
            comp_box, from_=10, to=100, variable=self.var_quality, orient=tk.HORIZONTAL, length=130
        )
        self.slider_quality.grid(row=0, column=1, padx=4)
        self.lbl_quality_val = ttk.Label(comp_box, textvariable=self.var_quality, width=4)
        self.lbl_quality_val.grid(row=0, column=2, padx=2)

        self.rb_target = ttk.Radiobutton(
            comp_box, text="Batasi Ukuran Target (KB):", variable=self.var_comp_mode,
            value="target_kb", command=self._toggle_compression_inputs
        )
        self.rb_target.grid(row=0, column=3, sticky=tk.W, padx=(20, 4))

        self.var_target_kb = tk.StringVar(value="300")
        self.entry_target_kb = ttk.Entry(comp_box, textvariable=self.var_target_kb, width=8, state=tk.DISABLED)
        self.entry_target_kb.grid(row=0, column=4, padx=4)
        ttk.Label(comp_box, text="KB").grid(row=0, column=5)

        ttk.Separator(opt_group, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        # Resize Baris
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

        # 4. Lokasi Penyimpanan Hasil
        dst_group = ttk.LabelFrame(main_frame, text=" 4. Lokasi Penyimpanan Hasil ", padding=10)
        dst_group.pack(fill=tk.X, pady=4)

        self.var_same_dir = tk.BooleanVar(value=True)
        chk_same = ttk.Checkbutton(
            dst_group, text="Simpan di folder yang sama dengan file asli (dengan imbuhan '_gray')",
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

        # 5. Tombol Aksi & Progress
        act_box = ttk.Frame(main_frame, padding=(0, 6))
        act_box.pack(fill=tk.X)

        self.btn_convert = ttk.Button(act_box, text="Mulai Konversi ke Grayscale", command=self._start_conversion_thread)
        self.btn_convert.pack(fill=tk.X, ipady=5)

        self.prog_bar = ttk.Progressbar(main_frame, orient=tk.HORIZONTAL, mode="determinate")
        self.prog_bar.pack(fill=tk.X, pady=(6, 4))

        self.lbl_status = ttk.Label(main_frame, text="Siap.", foreground="#475569")
        self.lbl_status.pack(anchor=tk.W)

        # 6. Log Console
        log_group = ttk.LabelFrame(main_frame, text=" Log Aktivitas ", padding=6)
        log_group.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

        self.log_text = ScrolledText(log_group, height=6, font=("Consolas", 9), wrap=tk.WORD, bg="#0f172a", fg="#f8fafc")
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _on_format_changed(self):
        fmt = self.var_out_format.get().lower()
        if fmt != "jpg":
            # Kompresi target KB hanya tersedia untuk JPG
            self.var_comp_mode.set("quality")
            self.rb_target.config(state=tk.DISABLED)
            self.entry_target_kb.config(state=tk.DISABLED)
            if fmt == "png":
                self.rb_quality.config(text="Lossless (Default PNG)")
                self.slider_quality.config(state=tk.DISABLED)
            else:
                self.rb_quality.config(text="Tanpa Kompresi (BMP)")
                self.slider_quality.config(state=tk.DISABLED)
        else:
            self.rb_target.config(state=tk.NORMAL)
            self.rb_quality.config(text="Kualitas Manual (1-100):")
            self._toggle_compression_inputs()

    def _append_log(self, text: str):
        self.log_text.insert(tk.END, text + "\n")
        self.log_text.see(tk.END)

    def _toggle_compression_inputs(self):
        if self.var_out_format.get().lower() != "jpg":
            return
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
            title="Pilih Gambar Warna",
            filetypes=[
                ("Citra Terbaca", "*.jpg;*.jpeg;*.png;*.bmp;*.webp;*.tif;*.tiff"),
                ("Semua File", "*.*")
            ]
        )
        if files:
            self.selected_files = [Path(f) for f in files]
            self.lbl_file_count.config(text=f"{len(self.selected_files)} gambar terpilih.")
            self._append_log(f"[INFO] Dipilih {len(self.selected_files)} file gambar.")

    def _select_folder(self):
        folder = filedialog.askdirectory(title="Pilih Folder Berisi Gambar")
        if folder:
            dir_path = Path(folder)
            found = [f for f in dir_path.glob("*.*") if f.suffix.lower() in SUPPORTED_EXTENSIONS]
            if not found:
                messagebox.showwarning("Peringatan", "Tidak ditemukan file gambar yang didukung pada folder tersebut.")
                return
            self.selected_files = found
            self.lbl_file_count.config(text=f"{len(self.selected_files)} gambar ditemukan di '{dir_path.name}'.")
            self._append_log(f"[INFO] Ditemukan {len(self.selected_files)} file gambar di folder {folder}.")

    def _select_output_folder(self):
        folder = filedialog.askdirectory(title="Pilih Folder Penyimpanan Output")
        if folder:
            self.output_directory = folder
            self.var_output_path.set(folder)

    def _clear_files(self):
        self.selected_files = []
        self.lbl_file_count.config(text="Belum ada gambar yang dipilih.")
        self.prog_bar["value"] = 0
        self.lbl_status.config(text="Siap.")

    def _start_conversion_thread(self):
        if not self.selected_files:
            messagebox.showwarning("Peringatan", "Silakan pilih setidaknya satu gambar terlebih dahulu!")
            return

        if not self.var_same_dir.get() and not self.var_output_path.get().strip():
            messagebox.showwarning("Peringatan", "Silakan tentukan folder tujuan penyimpanan!")
            return

        self.btn_convert.config(state=tk.DISABLED)
        self.is_processing = True

        thread = threading.Thread(target=self._process_conversion, daemon=True)
        thread.start()

    def _process_conversion(self):
        total_files = len(self.selected_files)
        success_count = 0
        method_name = self.combo_method.get()
        method_key = self._method_map.get(method_name, "luminance_bt601")
        out_fmt = self.var_out_format.get().lower()

        comp_mode = self.var_comp_mode.get()
        target_kb = 300
        manual_quality = self.var_quality.get()

        if comp_mode == "target_kb" and out_fmt == "jpg":
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

        self._append_log("\n--- Memulai Proses Konversi ke Grayscale ---")
        self._append_log(f"Metode: {method_name} | Format: {out_fmt.upper()}")

        for idx, src_path in enumerate(self.selected_files, start=1):
            try:
                if not src_path.exists():
                    self._append_log(f"[GAGAL] File '{src_path.name}' tidak ditemukan.")
                    continue

                # Nama file output
                new_ext = f".{out_fmt}"
                if save_to_same:
                    out_path = src_path.with_name(f"{src_path.stem}_gray{new_ext}")
                else:
                    out_path = custom_out_dir / f"{src_path.stem}_gray{new_ext}"

                with Image.open(src_path) as img:
                    orig_size_kb = src_path.stat().st_size / 1024
                    
                    # 1. Konversi ke Grayscale
                    gray_img = convert_to_grayscale(img, method_key)

                    # 2. Resize jika diminta
                    final_img = resize_image(gray_img, resize_mode, scale_val, max_w, max_h)

                    # 3. Simpan sesuai format & opsi kompresi
                    if out_fmt == "jpg":
                        if comp_mode == "target_kb":
                            final_q, final_kb = compress_to_target_kb(final_img, out_path, target_kb)
                            log_msg = f"[OK] {src_path.name} -> {out_path.name} ({orig_size_kb:.1f} KB -> {final_kb:.1f} KB, Q={final_q})"
                        else:
                            final_img.save(out_path, format="JPEG", quality=manual_quality, optimize=True)
                            final_kb = out_path.stat().st_size / 1024
                            log_msg = f"[OK] {src_path.name} -> {out_path.name} ({orig_size_kb:.1f} KB -> {final_kb:.1f} KB, Q={manual_quality})"
                    elif out_fmt == "png":
                        final_img.save(out_path, format="PNG", optimize=True)
                        final_kb = out_path.stat().st_size / 1024
                        log_msg = f"[OK] {src_path.name} -> {out_path.name} ({orig_size_kb:.1f} KB -> {final_kb:.1f} KB)"
                    else:  # BMP
                        final_img.save(out_path, format="BMP")
                        final_kb = out_path.stat().st_size / 1024
                        log_msg = f"[OK] {src_path.name} -> {out_path.name} ({orig_size_kb:.1f} KB -> {final_kb:.1f} KB)"

                    self._append_log(log_msg)
                    success_count += 1

            except Exception as e:
                self._append_log(f"[ERROR] Gagal memproses {src_path.name}: {e}")

            self.prog_bar["value"] = idx
            self.lbl_status.config(text=f"Memproses {idx}/{total_files} file...")

        self.lbl_status.config(text=f"Selesai: {success_count}/{total_files} citra berhasil diubah ke Grayscale.")
        self._append_log(f"--- Selesai: {success_count}/{total_files} berhasil dikonversi ---\n")
        self.btn_convert.config(state=tk.NORMAL)
        self.is_processing = False

        messagebox.showinfo(
            "Konversi Selesai",
            f"Proses selesai!\n{success_count} dari {total_files} gambar berhasil diubah ke Grayscale."
        )


def main():
    app = GrayscaleConverterGUI()
    app.mainloop()


if __name__ == "__main__":
    main()