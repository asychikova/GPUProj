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

    auto mpi_end = high_resolution_clock::now();
    auto communication_time_taken_ns = duration_cast<nanoseconds>(mpi_end - mpi_start).count();
  
    if (rank == 0) {
        long long total_compression_time_ns = accumulate(omp_times.begin(), omp_times.end(), 0LL);
        long long total_time_ns = total_compression_time_ns + communication_time_taken_ns;

        // throughput Calculation
        double total_time_seconds = static_cast<double>(total_time_ns) / 1e9;
        double throughput = static_cast<double>(text.size()) / (total_time_seconds * 1e6); // MB/s

        //output Performance Metrics
        cout << "-------------------" << endl;
        cout << "Huffman Compression" << endl;
        cout << "-------------------" << endl;
        cout << "Total Compression Time: " << total_compression_time_ns << " nanoseconds" << endl;
        cout << "Total Communication Time: " << communication_time_taken_ns << " nanoseconds" << endl;
        cout << "Total Time (Compression + Communication): " << total_time_ns << " nanoseconds" << endl;
        cout << "Throughput: " << throughput << " MB/s" << endl;
    }


    MPI_Finalize();
    return 0;
}

