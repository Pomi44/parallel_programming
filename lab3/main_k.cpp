#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <numeric>

struct Matrix {
    int rows;
    int cols;
    std::vector<std::vector<int> > data;
};


Matrix generateRandomMatrix(int size) {
    Matrix m;
    m.rows = size;
    m.cols = size;
    m.data.resize(size);
    for (int i = 0; i < size; ++i) {
        m.data[i].resize(size);
        for (int j = 0; j < size; ++j) {
            m.data[i][j] = std::rand() % 101;
        }
    }
    return m;
}

void flatten(const Matrix& mat, std::vector<int>& flat) {
    flat.clear();
    for (size_t i = 0; i < mat.data.size(); ++i) {
        for (size_t j = 0; j < mat.data[i].size(); ++j) {
            flat.push_back(mat.data[i][j]);
        }
    }
}

Matrix unflatten(int rows, int cols, const std::vector<int>& flat) {
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.data.resize(rows);
    for (int i = 0; i < rows; ++i) {
        m.data[i].resize(cols);
    }
    for (int i = 0; i < rows * cols; ++i) {
        m.data[i / cols][i % cols] = flat[i];
    }
    return m;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    std::srand(static_cast<unsigned>(std::time(0)));
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<int> sizes;
    for (int i = 250; i <= 1500; i += 250) {
        sizes.push_back(i);
    }

    for (size_t idx = 0; idx < sizes.size(); ++idx) {
        int mat_size = sizes[idx];
        std::vector<double> timings;

        std::vector<std::vector<int> > all_A_flat(10), all_B_flat(10);
        if (rank == 0) {
            for (int i = 0; i < 10; ++i) {
                Matrix A = generateRandomMatrix(mat_size);
                Matrix B = generateRandomMatrix(mat_size);
                flatten(A, all_A_flat[i]);
                flatten(B, all_B_flat[i]);
            }
        }

        for (int set = 0; set < 10; ++set) {
            std::vector<int> flatA(mat_size * mat_size);
            std::vector<int> flatB(mat_size * mat_size);

            double start_time = 0.0;
            MPI_Bcast(&mat_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

            if (rank == 0) {
                flatA = all_A_flat[set];
                flatB = all_B_flat[set];
                start_time = MPI_Wtime();
            }

            MPI_Bcast(&flatA[0], mat_size * mat_size, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&flatB[0], mat_size * mat_size, MPI_INT, 0, MPI_COMM_WORLD);

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
            if (rank == 0) {
                final_result.resize(mat_size * mat_size);
            }

            std::vector<int> recvcounts(size), displs(size);
            for (int i = 0; i < size; ++i) {
                int rows = (i == size - 1) ? mat_size - i * rows_per_proc : rows_per_proc;
                recvcounts[i] = rows * mat_size;
                displs[i] = i * rows_per_proc * mat_size;
            }

            MPI_Gatherv(&partial_result[0], partial_result.size(), MPI_INT,
                        (rank == 0 ? &final_result[0] : 0),
                        &recvcounts[0], &displs[0], MPI_INT,
                        0, MPI_COMM_WORLD);

            if (rank == 0) {
                double end_time = MPI_Wtime();
                timings.push_back(end_time - start_time);
            }
        }

        if (rank == 0) {
            double sum = 0.0;
            for (size_t i = 0; i < timings.size(); ++i) {
                sum += timings[i];
            }
            double average = sum / timings.size();
            std::cout << "[MPI] Matrix size: " << mat_size
                      << " | Mean time for 10 tries: " << average << " s" << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}
