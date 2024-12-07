#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <mpi.h>
#include <numeric>
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

    //gather compressed data size from all processes to rank 0
    vector<int> chunkSizes(size);
    int localSize = localCompressed.size();
    MPI_Gather(&localSize, 1, MPI_INT, chunkSizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    vector<int> displs(size, 0);
    if (rank == 0) {
        for (int i = 1; i < size; ++i) {
            displs[i] = displs[i - 1] + chunkSizes[i - 1];
        }
    }

    //start communication timing
    double communication_start_time = MPI_Wtime();

    //gather compressed data from all processes to rank 0 using point-to-point communication
    vector<char> allCompressed;
    if (rank == 0) {
        int totalSize = displs[size - 1] + chunkSizes[size - 1];
        allCompressed.resize(totalSize);

        //copy rank 0's own data
        copy(localCompressed.begin(), localCompressed.end(), allCompressed.begin());

        //receive data from other ranks
        int offset = localSize;
        for (int i = 1; i < size; ++i) {
            MPI_Status status;
            MPI_Recv(allCompressed.data() + displs[i], chunkSizes[i], MPI_CHAR, i, 0, MPI_COMM_WORLD, &status);
        }
    } else {
        //send compressed data to rank 0
        MPI_Send(localCompressed.data(), localSize, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    }

    //end communication timing
    double communication_end_time = MPI_Wtime();
    double communication_time_ns = (communication_end_time - communication_start_time) * 1e9; // Convert to nanoseconds

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

    return omp_time_ns + communication_time_ns;
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

    double total_mpi_start_time = MPI_Wtime(); //start total MPI timing

    const string inputFile = "input4MB.txt";
    const string compressedFile = "rlecompressed.txt";
    const string decompressedFile = "rledecompressed.txt";

    //start compression timing
    double compression_start_time = MPI_Wtime();
    long long omp_time_ns = compressRLE(inputFile, compressedFile, rank, size);
    double compression_end_time = MPI_Wtime();

    //calculate compression time for the local process in nanoseconds
    double compression_time_ns = (compression_end_time - compression_start_time) * 1e9;

    //reduce compression times to calculate the total compression time across all processes
    double total_compression_time_ns = 0;
    MPI_Reduce(&compression_time_ns, &total_compression_time_ns, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    //synchronize all processes before starting communication timing
    MPI_Barrier(MPI_COMM_WORLD);

    //start communication timing
    double communication_start_time = MPI_Wtime();

    //gather compressed data size from all processes to rank 0 (communication happens here)
    vector<int> chunkSizes(size);
    int localSize = omp_time_ns;  //assuming omp_time_ns represents the compressed data size
    MPI_Gather(&localSize, 1, MPI_INT, chunkSizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    //synchronize all processes after communication
    MPI_Barrier(MPI_COMM_WORLD);

    //end communication timing
    double communication_end_time = MPI_Wtime();
    double communication_time_ns = (communication_end_time - communication_start_time) * 1e9; //convert to nanoseconds

    size_t text = 0;
    if (rank == 0) {
        text = accumulate(chunkSizes.begin(), chunkSizes.end(), 0UL);
    }

    double total_mpi_end_time = MPI_Wtime(); //end total MPI timing
    double total_time_ns = (total_mpi_end_time - total_mpi_start_time) * 1e9; //convert total time to nanoseconds
    decompressRLE(compressedFile, decompressedFile);
    if (rank == 0) {
        //totalSize holds the input size in bytes
        double total_time_ns = total_compression_time_ns + communication_time_ns;

        //throughput Calculation (MB/s)
        double total_time_seconds = static_cast<double>(total_time_ns) / 1e9;
        double throughput = (static_cast<double>(text) / total_time_seconds) / 1e6; // throughput in MB/s

        //display Performance Metrics
        cout << "-------------------" << endl;
        cout << "Run Length Encoding (RLE)" << endl;
        cout << "-------------------" << endl;
        cout << "Total Compression Time: " << total_compression_time_ns << " nanoseconds" << endl;
        cout << "Total Communication Time: " << communication_time_ns << " nanoseconds" << endl;
        cout << "Total Time (Compression + Communication): " << total_time_ns << " nanoseconds" << endl;
        cout << "Throughput: " << throughput << " MB/s" << endl;
    }

    MPI_Finalize(); //finalize MPI
    return 0;
}




