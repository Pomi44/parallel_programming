#include <iostream>
#include <filesystem>
#include <cstdlib>
#include <ctime>
#include <string>
#include <vector>
#include <fstream>
#include <windows.h>
#include "matrix_utils.hpp"

namespace fs = std::filesystem;

int main() {
    SetConsoleOutputCP(CP_UTF8);
    std::ios::sync_with_stdio(false);
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    std::vector<int> sizes = {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000};

    std::ofstream timing_file("../timings.txt");
    if (!timing_file.is_open()) {
        std::cerr << "Не удалось открыть файл timings.txt для записи.\n";
        return 1;
    }

    timing_file << "Размер\tСреднее время (сек)\n";

    for (int size : sizes) {
        std::string base_dir = "../matrices/" + std::to_string(size) + "x" + std::to_string(size);

        double total_time = 0.0;

        for (int set = 1; set <= 10; ++set) {
            std::string set_dir = base_dir + "/set_" + std::to_string(set);
            fs::create_directories(set_dir);

            Matrix A = generateRandomMatrix(size);
            Matrix B = generateRandomMatrix(size);

            double start = getCurrentTime();
            Matrix C = multiply(A, B);
            double end = getCurrentTime();
            double elapsed = end - start;
            total_time += elapsed;

            writeTXT(A, set_dir + "/A.txt");
            writeTXT(B, set_dir + "/B.txt");
            writeTXT(C, set_dir + "/C.txt");

            std::cout << "[" << size << "x" << size << " | сет " << set << "] "
                      << "Время: " << elapsed << " сек\n";
        }

        double average_time = total_time / 10.0;
        timing_file << size << "\t" << average_time << "\n";

        std::cout << "▶ Среднее время для " << size << "x" << size
                  << ": " << average_time << " сек\n\n";
    }

    timing_file.close();
    std::cout << "✅ Средние времена записаны в timings.txt\n";
    return 0;
}
