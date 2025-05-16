import numpy as np
import os
import time
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def read_matrix(path):
    with open(path, 'r') as f:
        return np.array([
            [int(num) for num in line.strip().split()]
            for line in f
        ])

root = "matrices"
log_lines = []
success = 0
total = 0

# Проверка матриц
for threads_folder in sorted(os.listdir(root)):
    threads_path = os.path.join(root, threads_folder)
    if not os.path.isdir(threads_path):
        continue

    for size_folder in sorted(os.listdir(threads_path)):
        size_path = os.path.join(threads_path, size_folder)
        if not os.path.isdir(size_path):
            continue

        for set_folder in sorted(os.listdir(size_path)):
            set_path = os.path.join(size_path, set_folder)
            if not os.path.isdir(set_path):
                continue

            A_path = os.path.join(set_path, "A.txt")
            B_path = os.path.join(set_path, "B.txt")
            C_path = os.path.join(set_path, "C.txt")

            if not all(os.path.exists(p) for p in [A_path, B_path, C_path]):
                log_lines.append(f"⚠️ Пропущен {threads_folder}/{size_folder}/{set_folder}: отсутствуют файлы.")
                continue

            try:
                A = read_matrix(A_path)
                B = read_matrix(B_path)
                C = read_matrix(C_path)

                expected = A @ B
                if np.array_equal(C, expected):
                    log_lines.append(f"✅ {threads_folder}/{size_folder}/{set_folder}: OK")
                    success += 1
                else:
                    max_diff = np.max(np.abs(C - expected))
                    log_lines.append(f"❌ {threads_folder}/{size_folder}/{set_folder}: ERROR (max diff: {max_diff})")
            except Exception as e:
                log_lines.append(f"⚠️ Ошибка в {threads_folder}/{size_folder}/{set_folder}: {e}")
            total += 1

# Лог проверок
with open("results.txt", "w", encoding="utf-8") as f:
    for line in log_lines:
        f.write(line + "\n")
    f.write(f"\nИТОГО: {success}/{total} матриц успешно проверены.\n")

# Чтение timings.txt и построение сравнения
df = pd.read_csv("timings.txt", sep='\t')

# Проверка нужных столбцов
required_columns = {'Потоки', 'Размер', 'Среднее время (сек)'}
if not required_columns.issubset(df.columns):
    print("❌ Ошибка: timings.txt должен содержать заголовки:", required_columns)
    print("Найдено:", df.columns.tolist())
    exit(1)

# Построение графика
plt.figure(figsize=(10, 6))
for threads in sorted(df['Потоки'].unique()):
    subset = df[df['Потоки'] == threads]
    plt.plot(subset['Размер'], subset['Среднее время (сек)'], marker='o', label=f"{threads} поток(ов)")

plt.title("Сравнение времени перемножения матриц при разном числе потоков")
plt.xlabel("Размер матрицы (NxN)")
plt.ylabel("Среднее время (сек)")
plt.grid(True)
plt.legend(title="Потоки")
plt.tight_layout()
plt.savefig("timing_comparison_plot.png")
plt.show()
