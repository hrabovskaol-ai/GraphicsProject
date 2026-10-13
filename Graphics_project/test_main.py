import unittest
import numpy as np

# Імпортуємо функції з основного файлу
from Graphics_project import calc_mse, otsu

class TestImageAlgorithms(unittest.TestCase):

    def test_calc_mse_3x3(self):
        """Тестування середньоквадратичної помилки на матриці 3x3."""
        # Створюємо оригінальне зображення 3x3 (усі пікселі = 10)
        original = np.array([
            [10, 10, 10],
            [20, 20, 20],
            [30, 30, 30]
        ], dtype=np.uint8)

        # Створюємо оброблене зображення: змінюємо лише ОДИН піксель на 10 одиниць
        processed = np.array([
            [10, 10, 10],
            [20, 20, 20],
            [30, 30, 40]  # 40 замість 30
        ], dtype=np.uint8)

        # Обчислюємо вручну: 
        # Квадрат різниці зміненого пікселя: (40 - 30)^2 = 100
        # Ділимо на загальну кількість пікселів 
        expected_mse = 100.0 / 9.0
        actual_mse = calc_mse(original, processed)

        # Перевіряємо, чи збігається результат функції з ручним розрахунком
        self.assertAlmostEqual(actual_mse, expected_mse, places=4)

    def test_otsu_realistic(self):
        """Тестування методу Оцу на матриці з двома групами яскравості."""
        arr = np.array([
            [40,  50,  45],
            [200, 210, 190],
            [220, 205, 55]
        ], dtype=float)

        threshold = otsu(arr)
        
        # Оскільки між 55 і 190 немає пікселів, метод знаходить ідеальний поділ
        # одразу після 55 (повертає ~55.17). Перевіряємо цей діапазон.
        self.assertTrue(50 < threshold < 60, f"Помилковий поріг: {threshold}")

if __name__ == "__main__":
    unittest.main()
