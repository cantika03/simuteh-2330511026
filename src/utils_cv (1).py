"""utils_cv.py: fungsi bantu umum SIMUTEH (dari notebook B1)."""
import cv2
import numpy as np
import matplotlib.pyplot as plt


def baca(path):
    """Membaca citra dan memberi pesan jika gagal."""
    img = cv2.imread(path)

    if img is None:
        raise FileNotFoundError(f"Citra tidak ditemukan: {path}")

    return img


def tampil(daftar, judul, baris=1, ukuran=(14, 4)):
    """Menampilkan beberapa citra berdampingan."""
    kolom = int(np.ceil(len(daftar) / baris))

    plt.figure(figsize=ukuran)

    for i, (im, jd) in enumerate(zip(daftar, judul)):
        plt.subplot(baris, kolom, i + 1)

        if im.ndim == 3:
            plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(im, cmap="gray", vmin=0, vmax=255)

        plt.title(jd)
        plt.axis("off")

    plt.tight_layout()


def kontras_bercak(kanal, mask_daun, mask_bercak):
    """Menghitung kontras bercak terhadap daun sehat."""
    k = kanal.astype(np.float64)

    sehat = (mask_daun > 0) & (mask_bercak == 0)
    bercak = mask_bercak > 0

    if bercak.sum() == 0:
        return float("nan")

    return abs(k[bercak].mean() - k[sehat].mean()) / (
        k[sehat].std() + 1e-6
    )


def statistik(gray):
    """Ringkasan statistik intensitas citra."""
    return {
        "rerata": round(float(gray.mean()), 1),
        "simpangan_baku": round(float(gray.std()), 1),
        "min": int(gray.min()),
        "maks": int(gray.max()),
        "persen_gelap": round(100 * float((gray < 50).mean()), 1),
        "persen_terang": round(100 * float((gray > 205).mean()), 1)
    }
