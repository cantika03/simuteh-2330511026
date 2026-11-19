# SIMUTEH — Praktikum Computer Vision

**Nama:** Cantika Indie Pahla  
**NIM:** 2330511026  
**SEED Dataset:** 1026

## Deskripsi Proyek
Proyek SIMUTEH (Sistem Pemeriksaan Mutu Daun Teh Berbasis Citra) merupakan rangkaian praktikum Computer Vision berkelanjutan untuk kelompok tani teh rakyat di Kabupaten Sukabumi. Repositori ini mencakup tahap awal hingga segmentasi: mulai dari pengenalan lingkungan, analisis warna, histogram, penapisan, dan praproses citra (Evaluasi P1–P4), dilanjutkan dengan deteksi tepi, operasi morfologi, analisis komponen terhubung, serta modul analisis bercak (Praktikum Lanjutan P5–P6)[cite: 6].

## Progres & Versi Tag
- `v0.1` — Akuisisi dan praproses citra (Evaluasi P1–P4)[cite: 6]
- `v0.2-tepi` — Eksperimen gradien, operator tepi, dan deteksi tepi Canny (`P5_tepi.ipynb`)[cite: 6]
- `v0.3-morfologi` — Operasi morfologi, analisis komponen terhubung, modul `src/bercak.py`, dan parameter `hasil/param_bercak.json` (`P6_morfologi.ipynb`)[cite: 6]

## Tujuan
1. Memahami dasar pengolahan citra digital, analisis warna, histogram, penapisan, dan praproses citra.
2. Membandingkan operator tepi dan menentukan parameter deteksi tepi Canny[cite: 6].
3. Menerapkan operasi morfologi, analisis komponen terhubung, dan penyaringan fitur bercak[cite: 6].
4. Menyusun modul analisis bercak (`src/bercak.py`) dan mengevaluasi ketepatan penentuan kelas mutu daun[cite: 6].

## Struktur Folder
- `data/sintetis/` — Dataset sintetis dan label citra (`label.csv`, `mask_daun`, `mask_bercak`)[cite: 6].
- `notebook/` — Notebook praktikum Computer Vision (`P5_tepi.ipynb`, `P6_morfologi.ipynb`, dan notebook tahap sebelumnya)[cite: 6].
- `src/` — Kode sumber Python (`praproses.py`, `utils_cv.py`, `utils_tepi.py`, `bercak.py`).
- `hasil/` — Hasil pengujian, analisis, dan `param_bercak.json`.
- `laporan/` — Laporan dan presentasi praktikum[cite: 6].
- `requirements.txt` — Daftar pustaka Python yang dibutuhkan.

## Teknologi yang Digunakan
- Python
- OpenCV
- NumPy
- Matplotlib
- scikit-image
- Pandas
- Pillow

## Cara Menjalankan
1. Unduh atau salin repository ini.
2. Instal pustaka yang diperlukan dengan perintah:

   `pip install -r requirements.txt`

3. Buka notebook praktikum menggunakan Google Colab atau Jupyter Notebook.
4. Sesuaikan lokasi dataset dan folder kerja dengan struktur folder proyek.
5. Jalankan sel notebook secara berurutan.

## Dataset
Dataset yang digunakan merupakan dataset sintetis (`SEED = 1026`, 60 citra berukuran 640 × 480 piksel: Sehat, Bercak Ringan, dan Bercak Berat pada lima kondisi cahaya)[cite: 6]. Dataset disimpan dalam folder `data/sintetis/`, yang mencakup citra, citra bersih, mask daun, mask bercak, serta file label[cite: 6].

## Catatan
Hasil pengujian digunakan untuk mempelajari penerapan metode pengolahan citra digital pada dataset sintetis[cite: 6].
