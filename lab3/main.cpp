#include <mpi.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <string>

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

void writeTXT(const Matrix& m, const std::string& filename) {
    std::ofstream file(filename);
    for (const auto& row : m.data) {
        for (size_t j = 0; j < row.size(); ++j) {
            file << row[j];
            if (j + 1 < row.size()) file << " ";
        }
        file << "\n";
    }
}

void flatten(const Matrix& mat, std::vector<int>& flat) {
    flat.clear();
    for (const auto& row : mat.data)
        flat.insert(flat.end(), row.begin(), row.end());
}

Matrix unflatten(int rows, int cols, const std::vector<int>& flat) {
    Matrix m{ rows, cols, {} };
    m.data.resize(rows, std::vector<int>(cols));
    for (int i = 0; i < rows * cols; ++i)
        m.data[i / cols][i % cols] = flat[i];
    return m;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    std::srand(static_cast<unsigned>(std::time(nullptr)));
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<int> sizes = { 10, 20, 30, 40, 50, 60 };
    std::ofstream timing_file;
    if (rank == 0) timing_file.open("timings_mpi.txt");

    for (int mat_size : sizes) {
        for (int set = 1; set <= 10; ++set) {
            Matrix A, B;
            std::vector<int> flatA, flatB;

            double start_time = 0.0;
            if (rank == 0) {
                A = generateRandomMatrix(mat_size);
                B = generateRandomMatrix(mat_size);
                flatten(A, flatA);
                flatten(B, flatB);
                start_time = MPI_Wtime();
            }

            MPI_Bcast(&mat_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (rank != 0) {
                flatA.resize(mat_size * mat_size);
                flatB.resize(mat_size * mat_size);
            }

            MPI_Bcast(flatA.data(), mat_size * mat_size, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(flatB.data(), mat_size * mat_size, MPI_INT, 0, MPI_COMM_WORLD);

            Matrix matA = unflatten(mat_size, mat_size, flatA);
            Matrix matB = unflatten(mat_size, mat_size, flatB);

            int rows_per_proc = mat_size / size;
            int start_row = rank * rows_per_proc;
            int end_row = (rank == size - 1) ? mat_size : start_row + rows_per_proc;

            std::vector<int> partial_result((end_row - start_row) * mat_size, 0);

            for (int i = start_row; i < end_row; ++i) {
                for (int j = 0; j < mat_size; ++j) {
                    for (int k = 0; k < mat_size; ++k) {
                        partial_result[(i - start_row) * mat_size + j] +=
                            matA.data[i][k] * matB.data[k][j];
                    }
                }
            }

            std::vector<int> final_result;
            if (rank == 0)
                final_result.resize(mat_size * mat_size);

            std::vector<int> recvcounts(size), displs(size);
            for (int i = 0; i < size; ++i) {
                int rows = (i == size - 1) ? mat_size - i * rows_per_proc : rows_per_proc;
                recvcounts[i] = rows * mat_size;
                displs[i] = i * rows_per_proc * mat_size;
            }

            MPI_Gatherv(partial_result.data(), partial_result.size(), MPI_INT,
                final_result.data(), recvcounts.data(), displs.data(), MPI_INT,
                0, MPI_COMM_WORLD);

            if (rank == 0) {
                double end_time = MPI_Wtime();
                double elapsed = end_time - start_time;

                Matrix C = unflatten(mat_size, mat_size, final_result);

                std::string set_dir = "./matrices/mpi/" + std::to_string(mat_size) + "x" + std::to_string(mat_size) + "/set_" + std::to_string(set);
                std::filesystem::create_directories(set_dir);

                writeTXT(A, set_dir + "/A.txt");
                writeTXT(B, set_dir + "/B.txt");
                writeTXT(C, set_dir + "/C.txt");

                std::cout << "[MPI | " << mat_size << " | set " << set << "] time: " << elapsed << " s\n";
                timing_file << size << "\t" << mat_size << "\t" << elapsed << "\n";
            }
        }
    }

    if (rank == 0) {
        timing_file.close();
        std::cout << "��� ������� ������� �������� � timings_mpi.txt\n";
    }

    MPI_Finalize();
    return 0;
}
