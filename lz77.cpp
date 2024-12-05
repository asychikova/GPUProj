#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <mpi.h> 
#include "/usr/local/opt/libomp/include/omp.h" // OpenMP

using namespace std;

struct LZ77Tuple {
    int offset;
    int length;
    char nextChar;
};

//helper function to serialize LZ77Tuple into a vector of chars
vector<char> serialize(const vector<LZ77Tuple>& data) {
    vector<char> serialized_data;
    for (const auto& tuple : data) {
        serialized_data.insert(serialized_data.end(),
                               reinterpret_cast<const char*>(&tuple.offset),
                               reinterpret_cast<const char*>(&tuple.offset) + sizeof(int));
        serialized_data.insert(serialized_data.end(),
                               reinterpret_cast<const char*>(&tuple.length),
                               reinterpret_cast<const char*>(&tuple.length) + sizeof(int));
        serialized_data.push_back(tuple.nextChar);
    }
    return serialized_data;
}

//helper function to deserialize vector of chars into LZ77Tuple
vector<LZ77Tuple> deserialize(const vector<char>& data) {
    vector<LZ77Tuple> deserialized_data;
    size_t i = 0;
    while (i < data.size()) {
        LZ77Tuple tuple;
        tuple.offset = *reinterpret_cast<const int*>(&data[i]);
        i += sizeof(int);
        tuple.length = *reinterpret_cast<const int*>(&data[i]);
        i += sizeof(int);
        tuple.nextChar = data[i];
        i += sizeof(char);
        deserialized_data.push_back(tuple);
    }
    return deserialized_data;
}

//compress using LZ77 algorithm
vector<LZ77Tuple> lz77Compress(const string& input) {
    vector<LZ77Tuple> compressed;
    int windowSize = 4096;
    int bufferSize = 18;

    int pos = 0;
    while (pos < input.size()) {
        int maxLength = 0;
        int bestOffset = 0;
        char nextChar = input[pos];

        int start = max(0, pos - windowSize);

        //variables to store best match found by each thread
        int globalMaxLength = 0;
        int globalBestOffset = 0;
        char globalNextChar = input[pos];

        //parallel block to find the best match in the window
        #pragma omp parallel
        {
            int localMaxLength = 0;
            int localBestOffset = 0;
            char localNextChar = input[pos];

            #pragma omp for schedule(dynamic)
            for (int j = start; j < pos; ++j) {
                int length = 0;
                while (length < bufferSize && pos + length < input.size() && input[j + length] == input[pos + length]) {
                    ++length;
                }

                //update local variables for this thread
                if (length > localMaxLength) {
                    localMaxLength = length;
                    localBestOffset = pos - j;
                    if (pos + length < input.size()) {
                        localNextChar = input[pos + length];
                    }
                }
            }

            //update the global best match using a critical section
            #pragma omp critical
            {
                if (localMaxLength > globalMaxLength) {
                    globalMaxLength = localMaxLength;
                    globalBestOffset = localBestOffset;
                    globalNextChar = localNextChar;
                }
            }
        }

        //create and add the LZ77 tuple to the compressed data
        LZ77Tuple tuple = {globalBestOffset, globalMaxLength, globalNextChar};
        compressed.push_back(tuple);

        //move the position forward by the length of the match + 1 (or just 1 if no match)
        pos += (globalMaxLength > 0) ? globalMaxLength + 1 : 1;
    }

    return compressed;
}

//decompress using LZ77 algorithm
string lz77Decompress(const vector<LZ77Tuple>& compressed) {
    string decompressed;
    for (const auto& tuple : compressed) {
        if (tuple.offset == 0 && tuple.length == 0) {
            //add the next character as it is
            decompressed += tuple.nextChar;
        } else {
            //calculate start position for back reference
            int start = decompressed.size() - tuple.offset;
            if (start < 0) {
                cerr << "Error: Offset is larger than current decompressed size." << endl;
                continue;
            }

            //append the matching substring
            for (int j = 0; j < tuple.length; ++j) {
                decompressed += decompressed[start + j];
            }

            //add the next character after the matching substring
            decompressed += tuple.nextChar;
        }
    }
    return decompressed;
}


int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    //measure total MPI time
    double mpi_start_time = MPI_Wtime();

    ifstream inputFile("input1MB.txt");
    if (!inputFile) {
        cerr << "Error: Unable to open input file." << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    string input((istreambuf_iterator<char>(inputFile)), istreambuf_iterator<char>());
    inputFile.close();

    size_t segment_size = input.size() / world_size;
    string segment = input.substr(world_rank * segment_size, segment_size);

    //measure OpenMP time for compression
    double omp_start_time = omp_get_wtime();
    vector<LZ77Tuple> compressed_segment = lz77Compress(segment);
    double omp_end_time = omp_get_wtime();

    //report the time taken for compression by each process
    double omp_time_taken = omp_end_time - omp_start_time;
    cout << "Process " << world_rank << " - OpenMP Compression Time: " << omp_time_taken * 1e9 << " nanoseconds" << endl;

    vector<char> serialized_segment = serialize(compressed_segment);
    int local_size = serialized_segment.size();

    vector<int> segment_lengths(world_size);
    MPI_Gather(&local_size, 1, MPI_INT, segment_lengths.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    vector<char> gathered_data;
    if (world_rank == 0) {
        int total_size = 0;
        for (int len : segment_lengths) {
            total_size += len;
        }
        gathered_data.resize(total_size);

        copy(serialized_segment.begin(), serialized_segment.end(), gathered_data.begin());

        int offset = local_size;
        for (int i = 1; i < world_size; ++i) {
            MPI_Recv(gathered_data.data() + offset, segment_lengths[i], MPI_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            offset += segment_lengths[i];
        }
    } else {
        MPI_Send(serialized_segment.data(), local_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    }

    if (world_rank == 0) {
        vector<LZ77Tuple> final_compressed = deserialize(gathered_data);

        ofstream compressedFile("lz77compressed.txt", ios::binary);
        for (const auto& tuple : final_compressed) {
            compressedFile.write(reinterpret_cast<const char*>(&tuple), sizeof(LZ77Tuple));
        }
        compressedFile.close();

        string decompressed = lz77Decompress(final_compressed);
        ofstream decompressedFile("lz77decompressed.txt");
        decompressedFile << decompressed;
        decompressedFile.close();

        double total_omp_time = 0;
        for (int i = 0; i < world_size; ++i) {
            total_omp_time += omp_time_taken;
        }
        cout << "Total OpenMP Time: " << total_omp_time * 1e9 << " nanoseconds" << endl;

        double mpi_end_time = MPI_Wtime();
        double mpi_time_taken = mpi_end_time - mpi_start_time;
        cout << "Total MPI Time: " << mpi_time_taken * 1e9 << " nanoseconds" << endl;
        cout << "Total Time (MPI + OpenMP): " << (total_omp_time + mpi_time_taken) * 1e9 << " nanoseconds" << endl;
    }

    MPI_Finalize();
    return 0;
}
