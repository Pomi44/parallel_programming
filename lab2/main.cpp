#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <windows.h>
#include <filesystem>

struct Matrix {
    int rows;
    int cols;
    std::vector<std::vector<int>> data;
};

Matrix generateRandomMatrix(int size) {
    Matrix m;
    m.rows = size;
    m.cols = size;
    m.data.resize(size, std::vector<int>(size));

    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
            m.data[i][j] = std::rand() % 101;

    return m;
}

Matrix multiply(const Matrix& A, const Matrix& B, int num_threads) {
    Matrix C;
    C.rows = A.rows;
    C.cols = B.cols;
    C.data.resize(C.rows, std::vector<int>(C.cols, 0));

    omp_set_num_threads(num_threads);

    #pragma omp parallel for
    for (int i = 0; i < A.rows; ++i)
        for (int j = 0; j < B.cols; ++j)
            for (int k = 0; k < A.cols; ++k)
                C.data[i][j] += A.data[i][k] * B.data[k][j];

    return C;
}

void writeTXT(const Matrix& m, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Ошибка при открытии файла " << filename << std::endl;
        return;
    }

    for (const auto& row : m.data) {
        for (size_t j = 0; j < row.size(); ++j) {
            file << row[j];
            if (j + 1 < row.size()) file << " ";
        }
        file << "\n";
    }
}

double getCurrentTime() {
    return static_cast<double>(clock()) / CLOCKS_PER_SEC;
}

int main() {
    SetConsoleOutputCP(CP_UTF8);
    std::ios::sync_with_stdio(false);
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    std::vector<int> sizes = {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000};
    std::vector<int> thread_counts = {1, 2, 4, 8, 16};

    std::ofstream timing_file("timings.txt");
    if (!timing_file.is_open()) {
        std::cerr << "Не удалось открыть файл timings.txt для записи.\n";
        return 1;
    }

    timing_file << "Потоки\tРазмер\tСреднее время (сек)\n";

    for (int threads : thread_counts) {
        std::cout << "=== Тестирование с " << threads << " потоками ===\n";

        for (int size : sizes) {
            std::string base_dir = "./matrices/" + std::to_string(threads) + "_threads/" + std::to_string(size) + "x" + std::to_string(size);
            double total_time = 0.0;

            for (int set = 1; set <= 10; ++set) {
                std::string set_dir = base_dir + "/set_" + std::to_string(set);
                std::filesystem::create_directories(set_dir);

                Matrix A = generateRandomMatrix(size);
                Matrix B = generateRandomMatrix(size);

                double start = getCurrentTime();
                Matrix C = multiply(A, B, threads);
                double end = getCurrentTime();
                double elapsed = end - start;
                total_time += elapsed;

                writeTXT(A, set_dir + "/A.txt");
                writeTXT(B, set_dir + "/B.txt");
                writeTXT(C, set_dir + "/C.txt");

                std::cout << "[" << threads << "пт | " << size << "x" << size << " | сет " << set << "] "
                          << "Время: " << elapsed << " сек\n";
            }

            double average_time = total_time / 10.0;
            timing_file << threads << "\t" << size << "\t" << average_time << "\n";

            std::cout << "▶ Среднее время для " << size << "x" << size
                      << " при " << threads << " потоках: " << average_time << " сек\n\n";
        }
    }

    timing_file.close();
    std::cout << "✅ Все средние времена записаны в timings.txt\n";
    return 0;
}
