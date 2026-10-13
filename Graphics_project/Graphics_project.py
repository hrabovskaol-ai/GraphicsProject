import json
import time
import argparse
import tkinter as tk
from pathlib import Path
from tkinter import filedialog
import numpy as np
from PIL import Image

# Глобальний словник для збереження конфігурації
CONFIG = {}

def load_config(config_path="config.json"):
    """Завантажує налаштування з JSON-файлу."""
    global CONFIG 
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            CONFIG = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Помилка завантаження {config_path}. Застосовано стандартні налаштування.")
        CONFIG = {
            "pipeline": "p1", "gsd_m": 0.3, "roi": None,
            "threshold": "auto", "eq_strength": 0.5, "filter_size": 5,
            "min_class_gap": 0.15, "fallback_threshold": 0.1
        }

def choose_with_dialogs():
    """Відкриває діалогові вікна для вибору файлів та папки."""
    files = filedialog.askopenfilenames(title="Оберіть зображення",filetypes=[("Зображення", "*.png *.jpg *.jpeg *.tif *.bmp"), ("Усі файли", "*.*")])
    folder = filedialog.askdirectory(title="Оберіть папку для результатів") if files else None
    return [Path(f) for f in files], (Path(folder) if folder else None)

def load_image(path):
    """Завантажує RGB зображення та перевіряє, чи воно чорно-біле (монохромне)."""
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    is_gray = bool(np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 1], rgb[..., 2])) # Перевірка на монохром (канали R, G та B однакові)
    return rgb, is_gray

def save(path, arr):
    """Нормалізує та зберігає масив як зображення."""
    arr = np.asarray(arr)
    if arr.dtype == bool:
        arr = arr.astype(np.uint8) * 255
    elif arr.dtype != np.uint8:
        lo, hi = arr.min(), arr.max()
        arr = ((arr - lo) / (hi - lo + 1e-12) * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)

def overlay(img, mask, alpha=0.5):
    """Створює візуалізацію: накладає маску на оригінал напівпрозорим червоним кольором."""
    out = img.astype(np.float64)
    
    out[mask] = (1 - alpha) * out[mask] + alpha * np.array([255.0, 0.0, 0.0]) # Змішуємо оригінал з червоним кольором [255, 0, 0] там, де mask == True
    return np.clip(np.floor(out + 0.5), 0, 255).astype(np.uint8)

#МЕТРИКИ 
def calc_mse(original, processed):
    """Середньоквадратична помилка."""
    diff = original.astype(float) - processed.astype(float)
    return float(np.mean(diff ** 2))

def calc_psnr(mse, max_val=255.0):
    """Пікове відношення сигнал/шум."""
    return float('inf') if mse == 0 else float(10 * np.log10((max_val ** 2) / mse))

def calc_ssim(first, second, size=7):
    """Спрощений індекс структурної подібності (SSIM). Обчислюється по кожному каналу окремо."""
    def box(arr):
        pad = size // 2
        padded = np.pad(arr, pad, mode="edge")
        res = np.zeros_like(arr, dtype=float)
        for i in range(size):
            for j in range(size):
                res += padded[i:i+arr.shape[0], j:j+arr.shape[1]]
        return res / (size * size)

    c1, c2 = (0.01 * 255)**2, (0.03 * 255)**2
    ssim_vals = []
    
    # Проходимось циклом по трьох каналах (0 - Червоний, 1 - Зелений, 2 - Синій)
    for chan in range(3):
        # Беремо лише один канал як чорно-білу 2D картинку
        f_chan = first[..., chan].astype(float)
        s_chan = second[..., chan].astype(float)
        
        m1, m2 = box(f_chan), box(s_chan)
        var1, var2 = box(f_chan**2) - m1**2, box(s_chan**2) - m2**2
        cov = box(f_chan * s_chan) - m1 * m2
        
        ssim_map = ((2 * m1 * m2 + c1) * (2 * cov + c2)) / ((m1**2 + m2**2 + c1) * (var1 + var2 + c2))
        ssim_vals.append(ssim_map.mean())
        
    # Повертаємо середнє значення SSIM для всього кольорового зображення
    return float(np.mean(ssim_vals))

#АЛГОРИТМИ
def otsu(a):
    """Класичний метод Оцу."""
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-9: return lo
    idx = ((a - lo) / (hi - lo) * 255).astype(int)
    h = np.bincount(idx.ravel(), minlength=256) / a.size
    w, m = h.cumsum(), (h * np.arange(256)).cumsum()
    d = w * (1 - w)
    d[d == 0] = np.inf
    k = ((m[-1] * w - m) ** 2 / d).argmax()
    return lo + (k + 0.5) / 255 * (hi - lo)

def choose_threshold(values):
    """Вибір порогу із захистом від однорідних картинок (лише ліс або лише місто)."""
    if CONFIG.get("threshold") != "auto":
        return float(CONFIG["threshold"])
    
    thr = otsu(values)
    low_class, high_class = values[values <= thr], values[values > thr]
    
    if low_class.size == 0 or high_class.size == 0:
        return CONFIG.get("fallback_threshold", 0.1)
    
    # Якщо різниця між знайденими класами занадто мала — Оцу помилився
    if high_class.mean() - low_class.mean() < CONFIG.get("min_class_gap", 0.15):
        return CONFIG.get("fallback_threshold", 0.1)
        
    return thr

# ----- ПАЙПЛАЙН 1 -----
def stage_histogram(s):
    """1.1 Гістограмне вирівнювання."""
    img = s["img"]
    v = img.max(axis=2)
    cdf = np.bincount(v.ravel(), minlength=256).cumsum()
    cmin = cdf[cdf > 0][0]
    
    if cdf[-1] > cmin:
        lut = np.round((cdf - cmin) / (cdf[-1] - cmin) * 255)
        nv = lut[v]
        img_eq = np.where(v[..., None] > 0, img * (nv / np.maximum(v, 1))[..., None], nv[..., None])
        eq_strength = CONFIG.get("eq_strength", 0.5)
        s["img_eq"] = (img.astype(float) * (1 - eq_strength) + img_eq * eq_strength).clip(0, 255).astype(np.uint8)
    else:
        s["img_eq"] = img.copy()
    return {"image": s["img_eq"]}

def stage_color(s):
    """1.2 Індекс ExG (з підтримкою монохромних фото)."""
    if s["is_gray"]:
        # Для ЧБ фото: що світліше, то більше герметизовано (використовуємо яскравість)
        s["index"] = 1.0 - s["img_eq"].max(axis=2) / 255.0
    else:
        x = s["img_eq"].astype(float) 
        t = x.sum(axis=2, keepdims=True)
        n = x / np.where(t == 0, 1, t)
        s["index"] = 2 * n[..., 1] - n[..., 0] - n[..., 2]
    return {"exg": s["index"]}

def stage_filter(s):
    """1.3 Box Filter."""
    idx = s["index"]
    k = CONFIG.get("filter_size", 5)
    pad = k // 2
    padded = np.pad(idx, pad, mode='edge')
    blurred = np.zeros_like(idx, dtype=float)
    for i in range(k):
        for j in range(k):
            blurred += padded[i:i+idx.shape[0], j:j+idx.shape[1]]
    s["index_blurred"] = blurred / (k * k)
    return {"exg_filtered": s["index_blurred"]}

def stage_segmentation(s):
    """1.4 Розумна сегментація з перевіркою Оцу."""
    thr = choose_threshold(s["index_blurred"])
    s["mask"] = s["index_blurred"] <= thr
    s["threshold"] = round(float(thr), 5)
    return {"mask": s["mask"]}

PIPELINES = {"p1": [("histogram", stage_histogram), ("color", stage_color),("filter", stage_filter), ("segmentation", stage_segmentation)]}

def run(name, img, is_gray, out_dir):
    """Запуск стадій для одного зображення."""
    state = {"img": img.copy(), "is_gray": is_gray}
    start = time.perf_counter()

    i = 1
    for stage, func in PIPELINES[name]:
        for key, arr in func(state).items():
            save(out_dir / f"{i}_{stage}_{key}.png", arr)
        i += 1

    # Зберігаємо красивий візуальний оверлей
    save(out_dir / "final_overlay.png", overlay(img, state["mask"]))

    elapsed = time.perf_counter() - start
    mse = calc_mse(img, state["img_eq"])
    sealed = state["mask"].sum()

    return {
        "pipeline": name,
        "threshold": state["threshold"],
        "mse": round(mse, 2),
        "psnr": round(calc_psnr(mse), 2),
        "ssim": round(calc_ssim(img, state["img_eq"]), 4),
        "sealed_pixels": int(sealed),
        "sealed_area_m2": round(sealed * (CONFIG.get("gsd_m", 0.3) ** 2), 2),
        "sealed_percent": round(100 * sealed / state["mask"].size, 2),
        "time_s": round(elapsed, 3)
    }

def main():
    """Точка входу з підтримкою командного рядка."""
    parser = argparse.ArgumentParser(description="Урбаністичний моніторинг")
    parser.add_argument("--input", help="Файл або папка з фото")
    parser.add_argument("--output", help="Папка для результатів")
    args = parser.parse_args()

    load_config()

    if args.input and args.output:
        path = Path(args.input)
        files = [path] if path.is_file() else list(path.glob("*.*"))
        out = Path(args.output)
    else:
        files, out = choose_with_dialogs()

    if not files or not out:
        print("Вхідні дані не обрано. Завершення.")
        return

    print("Починаємо обробку...")
    rows = []
    roi = CONFIG.get("roi")

    for f in files:
        try:
            img, is_gray = load_image(f)
            if roi:
                img = img[max(roi[1], 0):roi[1]+roi[3], max(roi[0], 0):roi[0]+roi[2]]
            
            out_d = out / "p1" / f.stem
            out_d.mkdir(parents=True, exist_ok=True)
            
            row = {"image": f.name, **run("p1", img, is_gray, out_d)}
            rows.append(row)
            print(f"{f.name}: Забудова {row['sealed_percent']}% | SSIM: {row['ssim']}")
            
        except (OSError, ValueError) as err:
            print(f"[ПРОПУЩЕНО] {f.name}: {err}")

    if rows:
        with open(out / "metrics.txt", "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(f"=== {r['image']} ===\n")
                for k, v in r.items():
                    if k != 'image': fh.write(f"{k.upper()}: {v}\n")
                fh.write("\n")
        print(f"\nГотово. Результати у: {out}")

if __name__ == "__main__":
    main()