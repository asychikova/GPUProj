#include <mpi.h>
#include <iostream>
#include <fstream>
#include <queue>
#include <unordered_map>
#include <vector>
#include <string>
#include <memory>
#include <numeric>
#include "/usr/local/opt/libomp/include/omp.h" // OpenMP
#include <chrono>

using namespace std;
using namespace std::chrono;

//node structure for Huffman Tree
struct Node {
    char character;
    int frequency;
    unique_ptr<Node> left, right;

    Node(char ch, int freq, Node* l = nullptr, Node* r = nullptr)
        : character(ch), frequency(freq), left(l), right(r) {}
};

//comparator for priority queue
struct Compare {
    bool operator()(const Node* a, const Node* b) {
        return a->frequency > b->frequency;
    }
};

//function to build frequency map
unordered_map<char, int> buildFrequencyMap(const string& text) {
    unordered_map<char, int> freqMap;
    for (size_t i = 0; i < text.size(); ++i) {
        freqMap[text[i]]++;
    }
    return freqMap;
}

//function to build Huffman Tree
Node* buildHuffmanTree(const unordered_map<char, int>& freqMap) {
    priority_queue<Node*, vector<Node*>, Compare> pq;

    for (const auto& entry : freqMap) {
        pq.push(new Node(entry.first, entry.second));
    }

    while (pq.size() > 1) {
        Node* left = pq.top(); pq.pop();
        Node* right = pq.top(); pq.pop();
        pq.push(new Node('\0', left->frequency + right->frequency, left, right));
    }

    return pq.top();
}

//function to generate Huffman codes
void generateHuffmanCodes(Node* root, const string& code, unordered_map<char, string>& huffmanCodes) {
    if (!root) return;

    if (!root->left && !root->right) {
        huffmanCodes[root->character] = code;
    }

    generateHuffmanCodes(root->left.get(), code + "0", huffmanCodes);
    generateHuffmanCodes(root->right.get(), code + "1", huffmanCodes);
}

//function to compress the input text
string compressText(const string& text, const unordered_map<char, string>& huffmanCodes) {
    string compressed;
    #pragma omp parallel for reduction(+:compressed)
    for (size_t i = 0; i < text.size(); ++i) {
        #pragma omp critical
        compressed += huffmanCodes.at(text[i]);
    }
    return compressed;
}

//function to decompress the compressed text
string decompressText(const string& compressed, Node* root) {
    string decompressed;
    Node* current = root;
    for (char bit : compressed) {
        if (bit == '0') {
            current = current->left.get();
        } else {
            current = current->right.get();
        }

        if (!current->left && !current->right) {
            decompressed += current->character;
            current = root;
        }
    }
    return decompressed;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    auto mpi_start = high_resolution_clock::now();

    string text;
    if (rank == 0) {
        ifstream inputFile("inputHuffman4MB.txt");
        if (!inputFile) {
            cerr << "Error: Could not open input.txt" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        text.assign((istreambuf_iterator<char>(inputFile)), istreambuf_iterator<char>());
        inputFile.close();

        if (text.empty()) {
            cerr << "Error: Input file is empty." << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    //broadcast text size to all processes
    int textSize = text.size();
    MPI_Bcast(&textSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

    //distribute text chunks
    int chunkSize = (textSize + size - 1) / size;
    vector<char> localText(chunkSize, '\0');
    MPI_Scatter(rank == 0 ? text.data() : nullptr, chunkSize, MPI_CHAR, localText.data(), chunkSize, MPI_CHAR, 0, MPI_COMM_WORLD);

    auto omp_start = high_resolution_clock::now();
    //compute local frequency map
    unordered_map<char, int> localFreqMap = buildFrequencyMap(string(localText.begin(), localText.end()));
    auto omp_end = high_resolution_clock::now();
    auto omp_duration = duration_cast<nanoseconds>(omp_end - omp_start).count();
    

    //gather OpenMP times at root process
    vector<long long> omp_times(size);
    MPI_Gather(&omp_duration, 1, MPI_LONG_LONG, omp_times.data(), 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    //convert local frequency map to array format for MPI_Reduce
    int freqArray[256] = {0};
    for (const auto& entry : localFreqMap) {
        freqArray[static_cast<unsigned char>(entry.first)] = entry.second;
    }

    int globalFreqArray[256] = {0};
    MPI_Reduce(freqArray, globalFreqArray, 256, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        cout << "----------------" << endl;
        cout << "Huffman Encoding" << endl;
        cout << "----------------" << endl;
        //output OpenMP times for all processes
        long long total_omp_time = 0;
        for (int i = 0; i < size; ++i) {
            cout << "Process " << i << " - OpenMP Compression Time: " << omp_times[i] << " nanoseconds" << endl;
            total_omp_time += omp_times[i];
        }
        cout << "Total OpenMP Time: " << total_omp_time << " nanoseconds" << endl;

        //build Huffman Tree on root process
        unordered_map<char, int> globalFreqMap;
        for (int i = 0; i < 256; ++i) {
            if (globalFreqArray[i] > 0) {
                globalFreqMap[static_cast<char>(i)] = globalFreqArray[i];
            }
        }

        unique_ptr<Node> root(buildHuffmanTree(globalFreqMap));

        //generate Huffman codes
        unordered_map<char, string> huffmanCodes;
        generateHuffmanCodes(root.get(), "", huffmanCodes);

        auto omp_compress_start = high_resolution_clock::now();
        //compress text
        string compressed = compressText(text, huffmanCodes);
        auto omp_compress_end = high_resolution_clock::now();
        auto omp_compress_duration = duration_cast<nanoseconds>(omp_compress_end - omp_compress_start).count();
        

        //write compressed text to file
        ofstream compressedFile("huffman_compressed.txt");
        compressedFile << compressed;
        compressedFile.close();

        //decompress text
        string decompressed = decompressText(compressed, root.get());

        //write decompressed text to file
        ofstream decompressedFile("huffman_decompressed.txt");
        decompressedFile << decompressed;
        decompressedFile.close();
    }

    auto mpi_end = high_resolution_clock::now();
    auto mpi_duration = duration_cast<nanoseconds>(mpi_end - mpi_start).count();
    if (rank == 0) {
        cout << "Total MPI Time: " << mpi_duration << " nanoseconds" << endl;
        cout << "Total Time (MPI + OpenMP): " << (mpi_duration + accumulate(omp_times.begin(), omp_times.end(), 0LL)) << " nanoseconds" << endl;
    }

    MPI_Finalize();
    return 0;
}
