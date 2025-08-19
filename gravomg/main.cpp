#include <fstream>
#include <iostream>

#include "MultigridSolverAPI.h"

#include<igl/readOBJ.h>

#include <unsupported/Eigen/SparseExtra>

Eigen::MatrixXi createNeighborMatrix(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F) {
    // Number of vertices
    int numVertices = V.rows();

    // Create a vector of sets to store neighbors
    std::vector<std::set<int>> neighbors(numVertices);

    // Populate the neighbor sets
    for (int f = 0; f < F.rows(); ++f) {
        // Each face has three vertices
        for (int i = 0; i < 3; ++i) {
            int v1 = F(f, i);
            for (int j = 0; j < 3; ++j) {
                if (i != j) { // Avoid adding self as neighbor
                    int v2 = F(f, j);
                    neighbors[v1].insert(v2); // Add v2 as neighbor of v1
                }
            }
        }
    }

    // Create the neighbor matrix
    int maxNeighbors = 0;
    for (const auto& neighborSet : neighbors) {
        maxNeighbors = std::max(maxNeighbors, static_cast<int>(neighborSet.size()));
    }

    Eigen::MatrixXi neigh(numVertices, maxNeighbors);

    // Fill the neighbor matrix
    for (int i = 0; i < numVertices; ++i) {
        int j = 0;
        for (int neighbor : neighbors[i]) {
            neigh(i, j++) = neighbor;
        }
        // Fill remaining entries with -1 if fewer neighbors than max
        for (; j < maxNeighbors; ++j) {
            neigh(i, j) = -1; // or some sentinel value
        }
    }

    return neigh;
}

Eigen::MatrixXd loadMatrixMarketArray(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::string line;
    // Skip comments/header lines
    while (std::getline(file, line)) {
        if (line[0] != '%') break;  // First non-comment line
    }

    std::istringstream iss(line);
    int rows, cols;
    if (!(iss >> rows >> cols)) {
        throw std::runtime_error("Invalid header in MatrixMarket file.");
    }

    Eigen::MatrixXd matrix(rows, cols);

    // Read matrix values col-wise
    for (int j = 0; j < cols; ++j) {
        for (int i = 0; i < rows; ++i) {
            double value;
            if (!(file >> value)) {
                throw std::runtime_error("Unexpected end of file.");
            }
            matrix(i, j) = value;
        }
    }

    return matrix;
}

void read_command_line(int argc, char* argv[])
{
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " <mesh.obj> <A.mtx> <B.mtx> <experiment_name> <timing.csv> [writeHeaders]\n";

    }

    std::string mesh_file = argv[1];
    std::string A_file = argv[2];
    std::string B_file = argv[3];
    std::string experiment_name = argv[4];
    std::string timing_file = argv[5];
    bool write_headers = (argc >= 7) ? (std::string(argv[6]) == "1" || std::string(argv[6]) == "true") : false;
}


int main(int argc, char* argv[]) 
{
    std::string mesh_file = "rxdragon.obj";
    std::string A_file = "dragon_A.mtx";
    std::string B_file = "dragon_B.mtx";
    std::string experiment_name = "Dragon_Test";
    std::string timing_file = "testtiming.csv";
    std::string input_directory = "";
    bool write_headers = false;

    Eigen::MatrixXd V;
    Eigen::MatrixXi F;


    if (argc >= 6)
    {
        mesh_file = argv[1];
        A_file = argv[2];
        B_file = argv[3];
        experiment_name = argv[4];
        timing_file = argv[5];
        write_headers = (argc >= 7) ? (std::string(argv[6]) == "1" || std::string(argv[6]) == "true") : false;
    }

    std::string filename = mesh_file;

    if (!igl::readOBJ(filename, V, F)) {
        std::cerr << "Error loading the mesh." << std::endl;
        return -1;
    }

    Eigen::MatrixXd B_d = loadMatrixMarketArray(input_directory + B_file);
    Eigen::SparseMatrix<double> A;

    // Load a .mtx file
    if (Eigen::loadMarket(A, input_directory + A_file)) {
        std::cout << "\nSuccessfully loaded the matrix A." << std::endl;
    }
    else {
        std::cerr << "\nFailed to load the matrix A." << std::endl;
        return 0;
    }
    A.makeCompressed();

    std::cout << "\n A Size: " << A.rows() << " x " << A.cols() << " with " << A.nonZeros() << " nnz" << std::endl;
    std::cout << "\n B Size: " << B_d.rows() << " x " << B_d.cols() << std::endl;

    //Gravo GMG
    Eigen::MatrixXi neigh = createNeighborMatrix(V, F);
    Eigen::SparseMatrix<double> M;
    igl::massmatrix(V, F, igl::MASSMATRIX_TYPE_BARYCENTRIC, M);

    double ratio = 8;
    int low_bound = 50;
    int cycle_type = 0;
    double tolerance = 1e-6;
    int stopping_criteria = 2;
    int pre_iters = 2;
    int post_iters = 2;
    int max_iter = 1000;
    bool check_voronoi = true;
    bool nested = false;
    Sampling sampling_strategy = FASTDISK;
    Weighting weighting = BARYCENTRIC;
    bool sig06 = false;
    Eigen::MatrixXd normals = V;
    bool verbose = false;
    bool debug = false;
    bool ablation = false;
    int ablation_num_points = 3;
    bool ablation_random = false;

    MultigridSolverAPI solver(
        V, neigh, M, //System
        ratio, low_bound,  // Hierarchy settings
        cycle_type, tolerance, stopping_criteria, pre_iters, post_iters, max_iter, // Solver settings
        check_voronoi, nested, sampling_strategy, weighting, sig06, normals, verbose, debug, // Debug settings
        ablation, ablation_num_points, ablation_random);// Ablations

    Eigen::MatrixXd X = solver.solve(A, B_d);

    if (!igl::writeOBJ("output.obj", X, F)) {
        std::cerr << "Wasn't able to write output";
    }

    solver.write_single_row_timing(experiment_name, timing_file, write_headers);
    return 0;
}
