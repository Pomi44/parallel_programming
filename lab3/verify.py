import numpy as np
import os
import time
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import glob
import re

def read_matrix(path):
    with open(path, 'r') as f:
        return np.array([
            [int(num) for num in line.strip().split()]
            for line in f
        ])


# === ПРОВЕРКА МАТРИЦ ===
root = "matrices"
log_lines = []
success = 0
total = 0

for size_folder in sorted(os.listdir(root)):
    size_path = os.path.join(root, size_folder)
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
            log_lines.append(f"⚠️ Пропущен {size_folder}/{set_folder}: отсутствуют файлы.")
            continue

        try:
            A = read_matrix(A_path)
            B = read_matrix(B_path)
            C = read_matrix(C_path)

            expected = A @ B
            if np.array_equal(C, expected):
                log_lines.append(f"✅ {size_folder}/{set_folder}: OK")
                print("+1")
                success += 1
            else:
                max_diff = np.max(np.abs(C - expected))
                log_lines.append(f"❌ {size_folder}/{set_folder}: ERROR (max diff: {max_diff})")
        except Exception as e:
            log_lines.append(f"⚠️ Ошибка в {size_folder}/{set_folder}: {e}")
        total += 1

with open("results.txt", "w", encoding="utf-8") as f:
    for line in log_lines:
        f.write(line + "\n")
    f.write(f"\nИТОГО: {success}/{total} матриц успешно проверены.\n")

# === ПОСТРОЕНИЕ ГРАФИКА ИЗ timings_mpi_*.txt ===

timing_files = glob.glob("timings_mpi_*.txt")
data = []

for file in timing_files:
    with open(file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                threads, size, t = parts
                data.append({
                    "Потоки": int(threads),
                    "Размер": int(size),
                    "Время": float(t)
                })

df = pd.DataFrame(data)
avg_df = df.groupby(["Потоки", "Размер"]).mean().reset_index()

plt.figure(figsize=(10, 6))
for threads in sorted(avg_df["Потоки"].unique()):
    subset = avg_df[avg_df["Потоки"] == threads]
    plt.plot(subset["Размер"], subset["Время"], marker='o', label=f"{threads} поток(ов)")

plt.title("Сравнение времени при разных потоках (MPI)")
plt.xlabel("Размер матрицы (NxN)")
plt.ylabel("Среднее время (сек)")
plt.grid(True)
plt.legend(title="Потоки")
plt.tight_layout()
plt.savefig("timing_mpi_comparison.png")
plt.show()

def plot_from_thread_files():
    thread_files = [
        "1_threads.txt", "2_threads.txt", "4_threads.txt",
        "8_threads.txt", "16_threads.txt", "20_threads.txt"
    ]

    all_data = []

    for filename in thread_files:
        thread_count = int(filename.split('_')[0])
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                match = re.search(r'Matrix size: (\d+) .*?: ([0-9.]+)', line)
                if match:
                    size = int(match.group(1))
                    time_sec = float(match.group(2))
                    all_data.append((thread_count, size, time_sec))

    # Группировка данных
    import pandas as pd
    df = pd.DataFrame(all_data, columns=["Потоки", "Размер", "Время"])

    plt.figure(figsize=(10, 6))
    for threads in sorted(df["Потоки"].unique()):
        subset = df[df["Потоки"] == threads]
        plt.plot(subset["Размер"], subset["Время"], marker='o', label=f"{threads} поток(ов)")

    plt.title("Зависимость времени от размера матрицы при разных потоках (MPI)")
    plt.xlabel("Размер матрицы (NxN)")
    plt.ylabel("Среднее время (сек)")
    plt.grid(True)
    plt.legend(title="Потоки")
    plt.tight_layout()
    plt.savefig("timing_threads_plot.png")
    plt.show()

# Вызов функции
plot_from_thread_files()