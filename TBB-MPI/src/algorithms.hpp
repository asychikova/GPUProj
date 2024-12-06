#ifndef ALGORITHMS_HPP
#define ALGORITHMS_HPP

#include <string>           // For std::string
#include <vector>           // For std::vector
#include <unordered_map>    // For std::unordered_map
#include <queue>            // For std::priority_queue
#include <tuple>            // For std::tuple
#include <utility>          // For std::pair
#include <mutex>            // For std::mutex
#include <fstream>          // For file I/O operations
#include <iostream>         // For standard I/O operations
#include <tbb/tbb.h>        // Intel Threading Building Blocks for parallelism

// ----------------------- LZ77 Structures and Functions -----------------------

/**
 * @struct LZ77Token
 * @brief Represents a single token in the LZ77 compression algorithm.
 *
 * The LZ77Token structure holds the necessary information to reconstruct the
 * original data during decompression. Each token contains:
 * - offset: The number of characters to look back from the current position.
 * - length: The number of characters to copy from the offset position.
 * - next: The next character following the copied sequence.
 */
struct LZ77Token {
    size_t offset; // Number of characters to look back from the current position
    size_t length; // Number of characters to copy from the offset position
    char next;     // Next character following the copied sequence
};

/**
 * @brief Compresses a chunk of data using the LZ77 algorithm.
 *
 * @param data The input string to be compressed.
 * @param window_size The size of the sliding window for searching matches (default: 20).
 * @return A vector of LZ77Token representing the compressed data.
 *
 * This function scans through the input data and generates LZ77 tokens by finding
 * the longest match within the sliding window. Each token indicates how many
 * characters to copy from a previous position and what character follows the
 * copied sequence.
 */
std::vector<LZ77Token> compress_lz77(const std::string& data, size_t window_size = 20);

/**
 * @brief Decompresses a chunk of LZ77 tokens to reconstruct the original data.
 *
 * @param tokens A vector of LZ77Token representing the compressed data.
 * @return The decompressed original string.
 *
 * This function iterates through the LZ77 tokens and reconstructs the original
 * data by copying sequences from previously decompressed data based on the
 * offset and length, and appending the next character.
 */
std::string decompress_lz77(const std::vector<LZ77Token>& tokens);

// ----------------------- Huffman Encoding Structures and Functions -----------------------

/**
 * @struct HuffmanNode
 * @brief Represents a node in the Huffman tree.
 *
 * The HuffmanNode structure is used to build the Huffman tree for encoding.
 * Each node contains:
 * - character: The character stored in the node (only for leaf nodes).
 * - frequency: The frequency of the character or the sum of frequencies in the subtree.
 * - left: Pointer to the left child node.
 * - right: Pointer to the right child node.
 */
struct HuffmanNode {
    char character;        // Character stored in the node (applicable for leaf nodes)
    int frequency;         // Frequency of the character or combined frequencies
    HuffmanNode* left;     // Pointer to the left child node
    HuffmanNode* right;    // Pointer to the right child node

    /**
     * @brief Constructor to initialize a HuffmanNode with a character and its frequency.
     *
     * @param ch The character to store in the node.
     * @param freq The frequency of the character.
     */
    HuffmanNode(char ch, int freq) : character(ch), frequency(freq), left(nullptr), right(nullptr) {}
};

/**
 * @struct CompareNode
 * @brief Comparator for the priority queue used in Huffman encoding.
 *
 * This comparator ensures that the HuffmanNode with the lowest frequency has
 * the highest priority in the priority queue.
 */
struct CompareNode {
    bool operator()(HuffmanNode* const& n1, HuffmanNode* const& n2) {
        return n1->frequency > n2->frequency; // Lower frequency nodes have higher priority
    }
};

/**
 * @brief Compresses a chunk of data using Huffman Encoding.
 *
 * @param data The input string to be compressed.
 * @return A vector of pairs where each pair consists of a character and its corresponding Huffman code.
 *
 * This function builds a Huffman tree based on the frequency of characters in the input data,
 * generates Huffman codes for each character, and encodes the data accordingly.
 */
std::vector<std::pair<char, std::string>> compress_huffman(const std::string& data);

/**
 * @brief Decompresses a chunk of Huffman tokens to reconstruct the original data.
 *
 * @param encoded_data A vector of pairs where each pair consists of a character and its corresponding Huffman code.
 * @return The decompressed original string.
 *
 * This function rebuilds the Huffman tree from the encoded data and decodes the Huffman codes
 * to reconstruct the original string.
 */
std::string decompress_huffman(const std::vector<std::pair<char, std::string>>& encoded_data);

// ----------------------- Run-Length Encoding Structures and Functions -----------------------

/**
 * @struct RLEToken
 * @brief Represents a single token in Run-Length Encoding (RLE).
 *
 * The RLEToken structure holds information about consecutive repeating characters.
 * Each token contains:
 * - character: The character being repeated.
 * - count: The number of consecutive repetitions of the character.
 */
struct RLEToken {
    char character;    // The character being repeated
    size_t count;      // Number of consecutive repetitions
};

/**
 * @brief Compresses a chunk of data using Run-Length Encoding (RLE).
 *
 * @param data The input string to be compressed.
 * @return A vector of RLEToken representing the compressed data.
 *
 * This function scans through the input data and creates RLE tokens by counting
 * consecutive occurrences of each character.
 */
std::vector<RLEToken> compress_rle(const std::string& data);

/**
 * @brief Decompresses a chunk of RLE tokens to reconstruct the original data.
 *
 * @param tokens A vector of RLEToken representing the compressed data.
 * @return The decompressed original string.
 *
 * This function iterates through the RLE tokens and reconstructs the original
 * data by repeating each character based on its count.
 */
std::string decompress_rle(const std::vector<RLEToken>& tokens);

// ----------------------- Utility Functions -----------------------

// -----------------------------------------------------------------------------
/**
 * @brief Reads a specific chunk of a file.
 *
 * @param filename The name of the file to read from.
 * @param start The starting byte position in the file.
 * @param end The ending byte position in the file.
 * @return A string containing the data read from the specified chunk.
 *
 * This function opens the specified file, seeks to the starting byte position,
 * and reads data up to the ending byte position. It handles cases where the
 * file cannot be opened or if the read operation fails.
 */
std::string read_file_chunk(const std::string& filename, std::size_t start, std::size_t end);

// -----------------------------------------------------------------------------
/**
 * @brief Template function to write compressed tokens to a file by appending.
 *
 * @tparam T The type of tokens to write (e.g., LZ77Token, RLEToken).
 * @param filename The base name of the file to write to.
 * @param tokens A vector of tokens to be written.
 * @param rank The rank of the MPI process (used to differentiate file names).
 *
 * This function serializes the provided tokens and writes them to a binary file.
 * The file name is constructed using the base filename and the process rank to
 * ensure uniqueness across multiple MPI processes.
 */
template <typename T>
void write_compressed_chunk(const std::string& filename, const std::vector<T>& tokens, int rank) {
    std::ofstream file;
    // Construct the file name with rank to avoid collisions
    file.open(filename + "_compressed_" + std::to_string(rank) + ".bin", std::ios::binary | std::ios::app);
    if (!file.is_open()) { // Check if the file was successfully opened
        std::cerr << "Error opening file for writing: " << filename << "\n";
        return; // Exit the function if the file cannot be opened
    }
    for (const auto& token : tokens) { // Iterate through each token
        file.write(reinterpret_cast<const char*>(&token), sizeof(token)); // Write the binary representation of the token
    }
    file.close(); // Close the file after writing
}

/**
 * @brief Template function to read compressed tokens from a file.
 *
 * @tparam T The type of tokens to read (e.g., LZ77Token, RLEToken).
 * @param filename The base name of the file to read from.
 * @param rank The rank of the MPI process (used to differentiate file names).
 * @return A vector of tokens read from the file.
 *
 * This function reads binary data from a file corresponding to the given rank,
 * deserializes it into tokens, and returns them in a vector. It handles cases
 * where the file cannot be opened.
 */
template <typename T>
std::vector<T> read_compressed_chunk(const std::string& filename, int rank) {
    std::vector<T> tokens; // Vector to store the read tokens
    // Construct the file name with rank to ensure correct file is read
    std::ifstream file(filename + "_compressed_" + std::to_string(rank) + ".bin", std::ios::binary);
    if (!file.is_open()) { // Check if the file was successfully opened
        std::cerr << "Error opening file for reading: " << filename << "\n";
        return tokens; // Return an empty vector if the file cannot be opened
    }
    T token; // Temporary variable to hold each token during reading
    while (file.read(reinterpret_cast<char*>(&token), sizeof(token))) { // Read tokens one by one
        tokens.push_back(token); // Add the token to the vector
    }
    file.close(); // Close the file after reading
    return tokens; // Return the vector of tokens
}

/**
 * @brief Writes decompressed data to a file by appending.
 *
 * @param filename The name of the file to write to.
 * @param data The decompressed data to be written.
 * @param rank The rank of the MPI process (used to differentiate file names).
 *
 * This function opens the specified file in append mode and writes the
 * decompressed data to it. The file name is constructed using the base
 * filename and the process rank to ensure uniqueness across multiple MPI
 * processes.
 */
void write_decompressed_file(const std::string& filename, const std::string& data, int rank);

#endif
