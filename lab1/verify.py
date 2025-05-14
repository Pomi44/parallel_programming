import numpy as np
import os
import time
from pathlib import Path
import matplotlib.pyplot as plt

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
timings = {}

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

            t_start = time.time()
            expected = A @ B
            t_end = time.time()

            matrix_size = int(size_folder.split("x")[0])
            if matrix_size not in timings:
                timings[matrix_size] = []
            timings[matrix_size].append(t_end - t_start)

            if np.array_equal(C, expected):
                log_lines.append(f"✅ {size_folder}/{set_folder}: OK")
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

timing_path = Path("timings.txt")

# Считываем файл
sizes = []
times = []

with timing_path.open(encoding="utf-8") as f:
    for line in f:
        if line.strip().startswith("Размер") or not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                size = int(parts[0])
                time_val = float(parts[1].replace(",", "."))
                sizes.append(size)
                times.append(time_val)
            except ValueError:
                continue

# Строим график
plt.figure(figsize=(10, 5))
plt.plot(sizes, times, marker='o')
plt.xlabel("Размер матрицы (NxN)")
plt.ylabel("Среднее время перемножения (сек)")
plt.title("Зависимость времени перемножения от размера матрицы")
plt.grid(True)
plt.tight_layout()
output_path = "timing_plot.png"
plt.savefig(output_path)
plt.show()
