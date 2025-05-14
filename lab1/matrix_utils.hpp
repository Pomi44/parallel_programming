#pragma once
#include <vector>
#include <string>

struct Matrix {
    int rows;
    int cols;
    std::vector<std::vector<int>> data;
};

Matrix generateRandomMatrix(int size);
Matrix multiply(const Matrix& A, const Matrix& B);
void writeTXT(const Matrix& m, const std::string& filename);
double getCurrentTime();
