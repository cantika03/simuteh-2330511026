
import cv2
import numpy as np

PARAM_BAWAAN = {
    "lebar": 640,
    "gamma": 1.0,
    "clahe_clip": 2.0,
    "clahe_tile": 8,
    "penapis": "median",
    "kernel": 3,
    "sigma": 1.0,
    "tajam_k": 0.0
}

def koreksi_gamma(img, gamma=1.0):
    gamma = max(float(gamma), 0.01)
    tabel = np.array([
        ((i / 255.0) ** (1.0 / gamma)) * 255
        for i in range(256)
    ]).clip(0, 255).astype(np.uint8)

    return cv2.LUT(img, tabel)

def praproses(img, parameter=None):
    p = PARAM_BAWAAN.copy()

    if parameter is not None:
        p.update(parameter)

    if img is None:
        raise ValueError("Citra tidak boleh kosong.")

    # 1. Resize dengan mempertahankan rasio
    tinggi, lebar = img.shape[:2]
    lebar_baru = int(p["lebar"])
    tinggi_baru = max(1, round(tinggi * lebar_baru / lebar))

    hasil = cv2.resize(
        img,
        (lebar_baru, tinggi_baru),
        interpolation=cv2.INTER_AREA
    )

    # 2. Koreksi gamma
    hasil = koreksi_gamma(hasil, p["gamma"])

    # 3. CLAHE pada kanal V di ruang warna HSV
    hsv = cv2.cvtColor(hasil, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    clahe = cv2.createCLAHE(
        clipLimit=float(p["clahe_clip"]),
        tileGridSize=(
            int(p["clahe_tile"]),
            int(p["clahe_tile"])
        )
    )

    v = clahe.apply(v)
    hsv = cv2.merge([h, s, v])
    hasil = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 4. Denoising
    penapis = p["penapis"].lower()
    kernel = int(p["kernel"])

    if kernel > 1 and kernel % 2 == 0:
        raise ValueError("Kernel harus ganjil.")

    if penapis == "median":
        if kernel > 1:
            hasil = cv2.medianBlur(hasil, kernel)

    elif penapis == "gaussian":
        hasil = cv2.GaussianBlur(
            hasil,
            (kernel, kernel),
            float(p["sigma"])
        )

    elif penapis == "bilateral":
        hasil = cv2.bilateralFilter(hasil, 9, 75, 75)

    else:
        raise ValueError("Pilihan penapis tidak dikenal.")

    # 5. Unsharp masking opsional
    tajam_k = float(p["tajam_k"])

    if tajam_k > 0:
        blur = cv2.GaussianBlur(hasil, (0, 0), 1.0)
        hasil = cv2.addWeighted(
            hasil, 1.0 + tajam_k,
            blur, -tajam_k, 0
        )

    return hasil
