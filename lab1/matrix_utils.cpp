#include "matrix_utils.hpp"
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <iostream>

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

Matrix multiply(const Matrix& A, const Matrix& B) {
    Matrix C;
    C.rows = A.rows;
    C.cols = B.cols;
    C.data.resize(C.rows, std::vector<int>(C.cols, 0));

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
