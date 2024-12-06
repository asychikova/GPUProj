#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <mpi.h>  
#include "/usr/local/opt/libomp/include/omp.h" // OpenMP

using namespace std;

string compressChunk(const string& chunk) {
    size_t n = chunk.size();
    string compressed = "";
    
    //variables to store the current character and its count
    char currentChar = chunk[0];
    int count = 1;

    //use OpenMP to parallelize the iteration over the chunk
    #pragma omp parallel for schedule(static) reduction(+:count) shared(compressed, currentChar)
    for (size_t i = 1; i < n; ++i) {
        //check if the current character is equal to the last one
        if (chunk[i] == currentChar) {
            count++;
        } else {
            //critical section to update the compressed string when switching characters
            #pragma omp critical
            {
                compressed += currentChar + to_string(count);
                currentChar = chunk[i];
                count = 1;
            }
        }
    }

    //handle the last character
    #pragma omp critical
    {
        compressed += currentChar + to_string(count);
    }
    return compressed;
}

string decompressChunk(const string& compressed) {
    string decompressed = "";
    size_t n = compressed.size();
    size_t i = 0;

    while (i < n) {
        char currentChar = compressed[i++];
        string countStr = "";

        while (i < n && isdigit(compressed[i])) {
            countStr += compressed[i++];
        }

        int count = stoi(countStr);
        decompressed += string(count, currentChar);
    }

    return decompressed;
}

long long compressRLE(const string& inputFile, const string& outputFile, int rank, int size) {
    ifstream input(inputFile, ios::binary);
    if (!input.is_open()) {
        cerr << "Error opening file for compression.\n";
        return -1;
    }

    string fileContent((istreambuf_iterator<char>(input)), istreambuf_iterator<char>());
    input.close();

    size_t totalSize = fileContent.size();
    size_t chunkSize = (totalSize + size - 1) / size; //divide data evenly across processes
    size_t start = rank * chunkSize;
    size_t end = min(start + chunkSize, totalSize);

    string localChunk = fileContent.substr(start, end - start);

    auto omp_start = chrono::high_resolution_clock::now(); //start OpenMP timing

    string localCompressed = compressChunk(localChunk); //compress the local chunk

    auto omp_end = chrono::high_resolution_clock::now(); //end OpenMP timing
    long long omp_time_ns = chrono::duration_cast<chrono::nanoseconds>(omp_end - omp_start).count();

    //gather all compressed results to rank 0
    vector<int> chunkSizes(size);
    int localSize = localCompressed.size();
    MPI_Gather(&localSize, 1, MPI_INT, chunkSizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    vector<int> displs(size, 0);
    if (rank == 0) {
        for (int i = 1; i < size; ++i) {
            displs[i] = displs[i - 1] + chunkSizes[i - 1];
        }
    }

    vector<char> allCompressed((rank == 0) ? displs[size - 1] + chunkSizes[size - 1] : 0);
    MPI_Gatherv(localCompressed.data(), localSize, MPI_CHAR, allCompressed.data(), chunkSizes.data(), displs.data(), MPI_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        //merge all compressed results into a single result
        string mergedCompressed = "";
        char lastChar = '\0';
        int lastCount = 0;

        size_t i = 0;
        while (i < allCompressed.size()) {
            char currentChar = allCompressed[i++];
            string countStr = "";

            while (i < allCompressed.size() && isdigit(allCompressed[i])) {
                countStr += allCompressed[i++];
            }

            int currentCount = stoi(countStr);

            if (currentChar == lastChar) {
                //merge counts for the same character
                lastCount += currentCount;
            } else {
                //append the previous character and its count to the result
                if (lastChar != '\0') {
                    mergedCompressed += lastChar + to_string(lastCount);
                }
                lastChar = currentChar;
                lastCount = currentCount;
            }
        }

        //append the final character and its count
        if (lastChar != '\0') {
            mergedCompressed += lastChar + to_string(lastCount);
        }

        //write the consolidated compressed result to the output file
        ofstream output(outputFile, ios::binary);
        if (!output.is_open()) {
            cerr << "Error writing to output file.\n";
            return -1;
        }
        output << mergedCompressed;
        output.close();
    }

    return omp_time_ns;
}

void decompressRLE(const string& inputFile, const string& outputFile) {
    ifstream input(inputFile, ios::binary);
    if (!input.is_open()) {
        cerr << "Error opening compressed file for decompression.\n";
        return;
    }

    string compressed((istreambuf_iterator<char>(input)), istreambuf_iterator<char>());
    input.close();

    string decompressed = decompressChunk(compressed);

    ofstream output(outputFile, ios::binary);
    if (!output.is_open()) {
        cerr << "Error opening decompressed file for writing.\n";
        return;
    }
    output << decompressed;
    output.close();
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv); //initialize MPI

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); //get the rank of the current process
    MPI_Comm_size(MPI_COMM_WORLD, &size); //get the total number of processes

    double mpi_start_time = MPI_Wtime(); //start MPI timing

    const string inputFile = "input1MB.txt";
    const string compressedFile = "rlecompressed.txt"; 
    const string decompressedFile = "rledecompressed.txt"; 

    long long omp_time_ns = compressRLE(inputFile, compressedFile, rank, size);

    if (rank == 0) {
        decompressRLE(compressedFile, decompressedFile);
    }

    double mpi_end_time = MPI_Wtime(); //end MPI timing
    double mpi_time_ns = (mpi_end_time - mpi_start_time) * 1e9; //convert MPI time to nanoseconds

    //reduce OpenMP times to calculate the total time
    long long total_omp_time_ns = 0;
    MPI_Reduce(&omp_time_ns, &total_omp_time_ns, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    //calculate total time for both MPI and OpenMP
    long long total_time_ns = 0;
    if (rank == 0) {
        total_time_ns = mpi_time_ns + total_omp_time_ns; //total MPI + OpenMP time
    }

    if (rank == 0) {
        cout << "-------------------" << endl;
        cout << "Run Length Encoding" << endl;
        cout << "-------------------" << endl;
        for (int i = 0; i < size; ++i) {
            cout << "Process " << i << " - OpenMP Compression Time: " << omp_time_ns << " nanoseconds\n";
        }
        cout << "Total OpenMP Time: " << total_omp_time_ns << " nanoseconds\n";
        cout << "Total MPI Time: " << mpi_time_ns << " nanoseconds\n";
        cout << "Total Time (MPI + OpenMP): " << total_time_ns << " nanoseconds\n";
    }

    MPI_Finalize(); //finalize MPI
    return 0;
}



