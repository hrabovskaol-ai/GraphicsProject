import csv
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog
import numpy as np
from PIL import Image

# НАЛАШТУВАННЯ 
PIPELINE = "p1"           
GSD = 0.3                 
ROI = None                
SEED = 42
THRESHOLD = "auto"        
EQ_STRENGTH = 0.5         # Сила гістограмного вирівнювання (0.0 - 1.0). 0.5 зберігає кольори
FILTER_SIZE = 5           # Розмір фільтра згладжування (5x5)


# вікна вибору 
def choose_images():  
    files = filedialog.askopenfilenames(title="Оберіть зображення",filetypes=[("Зображення", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"),("Усі файли", "*.*")])
    return [Path(f) for f in files]

def choose_output_folder():
    folder = filedialog.askdirectory(title="Оберіть папку для результатів")
    return Path(folder) if folder else None

# допоміжні функції 
def load(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)

def save(path, arr):
    arr = np.asarray(arr)
    if arr.dtype == bool:
        arr = arr.astype(np.uint8) * 255
    elif arr.dtype != np.uint8:
        lo, hi = arr.min(), arr.max()
        arr = ((arr - lo) / (hi - lo + 1e-12) * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)

def calc_mse(original, processed):
    return np.mean((original.astype(float) - processed.astype(float)) ** 2) #обчислює середньоквадратичну помилку (MSE) між пікселями оригінального та обробленого зображень

def calc_psnr(mse, max_val=255.0): #обчислює метрику пікового відношення сигнал/шум (PSNR) на основі MSE, щоб виміряти якість зображення після обробки, де вище значення вказує на меншу кількість спотворень т
    if mse == 0:
        return float('inf')
    return 10 * np.log10((max_val ** 2) / mse)

def otsu(a): #автоматично обчислити ідеальний поріг для бінаризації,
    lo, hi = a.min(), a.max() #Знаходимо мінімальне та максимальне значення в масиві a
    if hi - lo < 1e-9:#Розділяти нічого  різниця близька до 0
        return lo
    idx = ((a - lo) / (hi - lo) * 255).astype(int)
    h = np.bincount(idx.ravel(), minlength=256) / a.size #імовірність зустріти кожен колір
    w = h.cumsum()
    m = (h * np.arange(256)).cumsum()
    d = w * (1 - w)
    d[d == 0] = np.inf
    k = ((m[-1] * w - m) ** 2 / d).argmax()
    return lo + (k + 0.5) / 255 * (hi - lo)

#стадії пайплайна 1
def stage_histogram(s): #1.1 М'яке гістограмне вирівнювання 
    img = s["img"]
    v = img.max(axis=2)
    cdf = np.bincount(v.ravel(), minlength=256).cumsum()
    cmin = cdf[cdf > 0][0] #Знаходить найменше ненульове значення в гістограмі
    
    if cdf[-1] > cmin: #Перевіряє, чи не складається фото з одного суцільного кольору
        lut = np.round((cdf - cmin) / (cdf[-1] - cmin) * 255) #вирівнювання
        nv = lut[v] #нова, покращена яскравість пікселів.
        ratio = nv / np.maximum(v, 1) #Обчислює коефіцієнт того, у скільки разів змінилася яскравість кожного пікселя
        img_eq = np.where(v[..., None] > 0, img * ratio[..., None], nv[..., None]) #збільшимо контраст
        
        img_blended = img.astype(float) * (1 - EQ_STRENGTH) + img_eq * EQ_STRENGTH # Змішуємо оригінал з вирівняним, щоб не "випалити" кольори ґрунту
        img_final = img_blended.clip(0, 255).astype(np.uint8)
    else:
        img_final = img.copy()
        
    s["img_eq"] = img_final
    return {"image": img_final}

def stage_color(s):    # Індекс зеленості ExG.
    x = s["img_eq"].astype(float) 
    t = x.sum(axis=2, keepdims=True) #загальна інтенсивність
    n = x / np.where(t == 0, 1, t) #изначаючи, яку пропорцію становить кожен колір від загальної яскравості
    s["index"] = 2 * n[..., 1] - n[..., 0] - n[..., 2]
    return {"exg": s["index"]}

def stage_filter(s): #1.3 Прямокутний фільтр для видалення дрібного шуму
    idx = s["index"]
    k = FILTER_SIZE
    pad = k // 2 #ширинa рамки
    
    padded = np.pad(idx, pad, mode='edge')# Створюємо масив з відступами для крайових пікселів
    blurred = np.zeros_like(idx, dtype=float)
    
    for i in range(k): # Ручна згортка через зміщення вікна 
        for j in range(k):
            blurred += padded[i:i+idx.shape[0], j:j+idx.shape[1]]
            
    blurred /= (k * k)
    s["index_blurred"] = blurred
    return {"exg_filtered": blurred}

def stage_segmentation(s): # 1.4 Поріг Оцу (застосовується до очищеної від шуму карти).
    
    thr = otsu(s["index_blurred"]) if THRESHOLD == "auto" else float(THRESHOLD) # Рахуємо поріг вже по згладженій карті
    s["mask"] = s["index_blurred"] <= thr #порівнює кожен піксель зображення з порогом
    s["threshold"] = round(float(thr), 5) #Зберігає знайдене значення порогу у стан пайплайну, округливши його до 5 знаків після коми.
    return {"mask": s["mask"]}

# Реєстрація пайплайнів
PIPELINES = {"p1": [("histogram", stage_histogram),("color", stage_color),("filter", stage_filter),("segmentation", stage_segmentation)],}

def run(name, img, out_dir):
    state = {"img": img.copy()}
    start = time.perf_counter()

    i = 1
    for stage, func in PIPELINES[name]:
        for key, arr in func(state).items():
            save(out_dir / f"{i}_{stage}_{key}.png", arr)
        i += 1

    elapsed = time.perf_counter() - start

    mse = calc_mse(img, state["img_eq"]) #середньоквадратична помилка між оригіналом та обробленим зображенням.
    psnr = calc_psnr(mse) # показник співвідношення сигналу до шуму.
    sealed = state["mask"].sum() # кількість пікселів, які позначені маскою як запечатані поверхні.

    return {
        "pipeline": name,
        "threshold": state["threshold"],
        "mse": round(mse, 2),
        "psnr": round(psnr, 2),
        "sealed_pixels": int(sealed),
        "sealed_area_m2": round(sealed * GSD ** 2, 2),
        "sealed_percent": round(100 * sealed / state["mask"].size, 2),
        "time_s": round(elapsed, 3)
    }
# ГОЛОВНИЙ БЛОК
if __name__ == "__main__":
    files = choose_images()
    if not files:
        print("Зображення не обрано. Завершення.")
        raise SystemExit

    out = choose_output_folder()
    if out is None:
        print("Папку не обрано. Завершення.")
        raise SystemExit

    names = list(PIPELINES) if PIPELINE == "all" else [PIPELINE]
    rows = []
    
    print("Починаємо обробку...")
    for f in files:
        try:
            img = load(f)
            if ROI:
                x, y, w, h = ROI
                img = img[max(y, 0):y + h, max(x, 0):x + w]
            if img.size == 0:
                raise ValueError("порожнє зображення")
                
            for name in names:
                d = out / name / f.stem
                d.mkdir(parents=True, exist_ok=True)
                row = {"image": f.name, **run(name, img, d)}
                rows.append(row)
                print(f"{f.name} [{name}]: Забудова {row['sealed_percent']}% | PSNR: {row['psnr']}")
        except Exception as e:
            print(f"[ПРОПУЩЕНО] {f.name}: {e}")

    if rows: # Створюємо звичайний текстовий файл 
        with open(out / "metrics.txt", "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(f"Результати для зображення: {row['image']} \n")
                fh.write(f"Пайплайн: {row['pipeline']}\n")
                fh.write(f"Поріг Оцу: {row['threshold']}\n")
                fh.write(f"MSE: {row['mse']}\n")
                fh.write(f"PSNR: {row['psnr']}\n")
                fh.write(f"Пікселів забудови: {row['sealed_pixels']}\n")
                fh.write(f"Площа забудови: {row['sealed_area_m2']} м²\n")
                fh.write(f"Відсоток забудови: {row['sealed_percent']} %\n")
                fh.write(f"Час обробки: {row['time_s']} с\n")
                fh.write("-" * 40 + "\n\n") # Розділювач між зображеннями

        print(f"\nГотово. Усі проміжні результати збережено у: {out}")
        print(f"Метрики записано у файл: {out / 'metrics.txt'}")
    else:
        print("Жодне зображення не оброблено.")