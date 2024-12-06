#include "algorithms.hpp"  // Include the header file containing declarations
#include <tbb/tbb.h>        // Intel Threading Building Blocks for parallelism
#include <fstream>          // For file I/O operations
#include <iostream>         // For standard I/O operations
#include <algorithm>        // For algorithms like std::max
#include <cstring>          // For C-string functions

// ----------------------- LZ77 Implementation -----------------------

/**
 * @brief Compresses a chunk of data using the LZ77 algorithm.
 *
 * This function scans through the input data and generates LZ77 tokens by finding
 * the longest match within a sliding window. Each token contains:
 * - offset: Number of characters to look back from the current position.
 * - length: Number of characters to copy from the offset position.
 * - next: The next character following the copied sequence.
 *
 * @param data The input string to be compressed.
 * @param window_size The size of the sliding window for searching matches (default: 20).
 * @return A vector of LZ77Token representing the compressed data.
 */
std::vector<LZ77Token> compress_lz77(const std::string& data, size_t window_size) {
    std::vector<LZ77Token> tokens; // Vector to store the generated tokens
    size_t pos = 0;                 // Current position in the data
    size_t data_size = data.size(); // Total size of the input data

    while (pos < data_size) { // Loop until the end of the data is reached
        size_t match_length = 0;     // Length of the current match
        size_t match_distance = 0;   // Distance of the current match
        char next_char = data[pos];  // Next character after the match

        // Determine the start of the sliding window
        size_t window_start = (pos >= window_size) ? pos - window_size : 0;
        size_t window_length = pos - window_start;                     // Length of the sliding window
        std::string window = data.substr(window_start, window_length); // Extract the sliding window

        // Shared mutex for updating match_length and match_distance safely
        tbb::spin_mutex mutex;

        // Variables to hold the best match found by threads
        size_t local_match_length = 0;
        size_t local_match_distance = 0;
        char local_next_char = next_char;

        // Parallel search for the longest match using TBB
        tbb::parallel_for(tbb::blocked_range<size_t>(0, window.size()),
            [&](const tbb::blocked_range<size_t>& range) {
                size_t thread_local_length = 0;      // Local match length for the thread
                size_t thread_local_distance = 0;    // Local match distance for the thread
                char thread_local_next_char = next_char; // Local next character for the thread

                for (size_t i = range.begin(); i < range.end(); ++i) { // Iterate through the assigned range
                    size_t length = 0; // Initialize match length

                    // Ensure we don't exceed the window size or data size
                    while ((pos + length < data_size) &&
                        (i + length < window.size()) &&
                        (window[i + length] == data[pos + length])) {
                        length++; // Increment match length for each matching character
                    }

                    if (length > thread_local_length) { // Check if a longer match is found
                        thread_local_length = length;                    // Update local match length
                        thread_local_distance = window.size() - i;       // Update local match distance
                        if (pos + length < data_size) {                   // Ensure we don't go out of bounds
                            thread_local_next_char = data[pos + length];  // Update local next character
                        }
                        else {
                            thread_local_next_char = '\0'; // End of data reached
                        }
                    }
                }

                // Update the global match_length and match_distance if a better match is found
                if (thread_local_length > 0) { // If any match was found
                    tbb::spin_mutex::scoped_lock lock(mutex); // Lock the mutex to update shared variables
                    if (thread_local_length > local_match_length) { // Compare with the current best match
                        local_match_length = thread_local_length;     // Update the best match length
                        local_match_distance = thread_local_distance; // Update the best match distance
                        local_next_char = thread_local_next_char;     // Update the best next character
                    }
                }
            });

        if (local_match_length > 0) { // If a match was found
            tokens.emplace_back(LZ77Token{ local_match_distance, local_match_length, local_next_char }); // Add the token
            pos += local_match_length + 1; // Move past the matched characters and the next character
        }
        else { // No match found
            tokens.emplace_back(LZ77Token{ 0, 0, data[pos] }); // Add a token with no match
            pos += 1; // Move to the next character
        }
    }

    return tokens; // Return the vector of generated tokens
}

/**
 * @brief Decompresses a chunk of LZ77 tokens to reconstruct the original data.
 *
 * This function iterates through the LZ77 tokens and reconstructs the original
 * data by copying sequences from previously decompressed data based on the
 * offset and length, and appending the next character.
 *
 * @param tokens A vector of LZ77Token representing the compressed data.
 * @return The decompressed original string.
 */
std::string decompress_lz77(const std::vector<LZ77Token>& tokens) {
    std::string decompressed;             // String to store the decompressed data
    decompressed.reserve(tokens.size() * 2); // Reserve space to optimize memory allocations

    for (const auto& token : tokens) { // Iterate through each token
        if (token.offset > 0 && token.length > 0) { // If the token represents a match
            if (token.offset > decompressed.size()) { // Validate the offset
                std::cerr << "Invalid offset in LZ77 token.\n"; // Output error message
                continue; // Skip this token if offset is invalid
            }
            size_t start = decompressed.size() - token.offset; // Calculate the start position for copying
            for (size_t i = 0; i < token.length; ++i) { // Copy the matched sequence
                decompressed += decompressed[start + i]; // Append each character to the decompressed string
            }
        }
        if (token.next != '\0') { // If there is a next character
            decompressed += token.next; // Append the next character
        }
    }
    return decompressed; // Return the reconstructed original string
}

// ----------------------- Huffman Encoding Implementation -----------------------

/**
 * @brief Helper function to build a frequency table in parallel.
 *
 * This function divides the input data among multiple threads to count the frequency
 * of each character. Each thread maintains a local frequency table which is then
 * merged into a global frequency table.
 *
 * @param data The input string for which the frequency table is to be built.
 * @param freq_table The global frequency table to be populated.
 */
void build_frequency_table(const std::string& data, std::unordered_map<char, int>& freq_table) {
    // Initialize a vector of unordered_maps for local frequency tables, one per thread
    const size_t num_threads = tbb::this_task_arena::max_concurrency(); // Get the maximum number of concurrent threads
    std::vector<std::unordered_map<char, int>> local_freq_tables(num_threads); // Vector to store local frequency tables

    // Parallel loop to count frequencies
    tbb::parallel_for(tbb::blocked_range<size_t>(0, data.size()),
        [&](const tbb::blocked_range<size_t>& range) {
            size_t thread_id = tbb::this_task_arena::current_thread_index(); // Get the current thread's index
            if (thread_id >= local_freq_tables.size()) { // Ensure thread_id is within bounds
                thread_id = 0; // Fallback to the first table if out of bounds
            }
            auto& local_freq = local_freq_tables[thread_id]; // Reference to the current thread's local frequency table
            for (size_t i = range.begin(); i < range.end(); ++i) { // Iterate through the assigned range
                local_freq[data[i]]++; // Increment the frequency count for the character
            }
        });

    // Merge local frequency tables into the global frequency table
    for (const auto& local_freq : local_freq_tables) { // Iterate through each local frequency table
        for (const auto& pair : local_freq) { // Iterate through each character-frequency pair
            freq_table[pair.first] += pair.second; // Add the local frequency to the global frequency table
        }
    }
}

/**
 * @brief Builds the Huffman tree based on the frequency table.
 *
 * This function creates HuffmanNodes for each character and inserts them into a priority
 * queue. It then repeatedly removes the two nodes with the lowest frequency, merges them
 * into a new node, and reinserts the new node into the queue until only one node remains,
 * which becomes the root of the Huffman tree.
 *
 * @param freq_table A map containing character frequencies.
 * @return Pointer to the root of the Huffman tree.
 */
HuffmanNode* build_huffman_tree(const std::unordered_map<char, int>& freq_table) {
    std::priority_queue<HuffmanNode*, std::vector<HuffmanNode*>, CompareNode> pq; // Priority queue to build the Huffman tree

    for (const auto& pair : freq_table) { // Iterate through the frequency table
        pq.push(new HuffmanNode(pair.first, pair.second)); // Create and insert a new HuffmanNode into the queue
    }

    if (pq.empty()) { // Check if the priority queue is empty
        return nullptr; // Return nullptr if no nodes were added
    }

    while (pq.size() > 1) { // Continue until only one node remains in the queue
        HuffmanNode* left = pq.top(); pq.pop();  // Extract the node with the lowest frequency
        HuffmanNode* right = pq.top(); pq.pop(); // Extract the node with the next lowest frequency
        HuffmanNode* merged = new HuffmanNode('\0', left->frequency + right->frequency); // Create a new merged node
        merged->left = left;   // Assign the left child
        merged->right = right; // Assign the right child
        pq.push(merged);       // Insert the merged node back into the priority queue
    }

    return pq.top(); // The remaining node is the root of the Huffman tree
}

/**
 * @brief Generates Huffman codes by traversing the Huffman tree.
 *
 * This recursive function traverses the Huffman tree and assigns binary codes to each
 * character. Left traversal appends '0' to the current code, and right traversal appends '1'.
 *
 * @param root Pointer to the current node in the Huffman tree.
 * @param huffman_codes A map to store the generated Huffman codes for each character.
 * @param current_code The Huffman code accumulated so far during traversal.
 */
void generate_codes(HuffmanNode* root, std::unordered_map<char, std::string>& huffman_codes, const std::string& current_code) {
    if (!root) // Base case: if the node is null, return
        return;
    if (!root->left && !root->right) { // If the node is a leaf (no children)
        huffman_codes[root->character] = current_code.empty() ? "0" : current_code; // Assign the current code to the character
        return;
    }
    generate_codes(root->left, huffman_codes, current_code + "0"); // Traverse the left subtree with '0' appended
    generate_codes(root->right, huffman_codes, current_code + "1"); // Traverse the right subtree with '1' appended
}

/**
 * @brief Compresses a chunk of data using Huffman Encoding.
 *
 * This function builds a frequency table, constructs the Huffman tree, generates Huffman
 * codes for each character, and encodes the data accordingly. The encoding process replaces
 * each character in the input data with its corresponding Huffman code.
 *
 * @param data The input string to be compressed.
 * @return A vector of pairs where each pair consists of a character and its corresponding Huffman code.
 */
std::vector<std::pair<char, std::string>> compress_huffman(const std::string& data) {
    std::unordered_map<char, int> freq_table; // Frequency table to store character frequencies
    // Build frequency table in parallel
    build_frequency_table(data, freq_table);

    // Build Huffman Tree
    HuffmanNode* root = build_huffman_tree(freq_table); // Construct the Huffman tree
    if (!root) { // Check if the Huffman tree was successfully built
        std::cerr << "Huffman tree could not be built. Data might be empty.\n"; // Output error message
        return {}; // Return an empty vector if the tree could not be built
    }

    // Generate Huffman Codes
    std::unordered_map<char, std::string> huffman_codes; // Map to store Huffman codes for each character
    generate_codes(root, huffman_codes, ""); // Generate codes by traversing the tree

    // Encode data
    std::vector<std::pair<char, std::string>> encoded_data; // Vector to store the encoded data

    // Shared mutex for updating encoded_data safely in parallel
    tbb::spin_mutex mutex;

    // Parallel loop to encode the data using Huffman codes
    tbb::parallel_for(tbb::blocked_range<size_t>(0, data.size()),
        [&](const tbb::blocked_range<size_t>& range) {
            std::vector<std::pair<char, std::string>> local_encoded; // Local vector to store encoded tokens
            local_encoded.reserve(range.size()); // Reserve space to optimize memory allocations
            for (size_t i = range.begin(); i < range.end(); ++i) { // Iterate through the assigned range
                local_encoded.emplace_back(std::make_pair(data[i], huffman_codes[data[i]])); // Encode each character
            }
            // Lock and append to encoded_data to avoid race conditions
            tbb::spin_mutex::scoped_lock lock(mutex);
            encoded_data.insert(encoded_data.end(), local_encoded.begin(), local_encoded.end()); // Merge local encoded data
        });
    return encoded_data; // Return the encoded data
}

/**
 * @brief Decompresses a chunk of Huffman tokens to reconstruct the original data.
 *
 * This function rebuilds the Huffman codes from the encoded data and decodes the
 * Huffman codes to reconstruct the original string.
 *
 * @param encoded_data A vector of pairs where each pair consists of a character and its corresponding Huffman code.
 * @return The decompressed original string.
 */
std::string decompress_huffman(const std::vector<std::pair<char, std::string>>& encoded_data) {
    // Reconstruct Huffman Codes
    std::unordered_map<std::string, char> reverse_codes; // Map to store reverse Huffman codes
    for (const auto& pair : encoded_data) { // Iterate through the encoded data
        reverse_codes[pair.second] = pair.first; // Populate the reverse codes map
    }

    // Decode data
    std::string decompressed;       // String to store the decompressed data
    std::string current_code;       // Variable to accumulate the current Huffman code
    decompressed.reserve(encoded_data.size() * 2); // Reserve space to optimize memory allocations

    for (const auto& pair : encoded_data) { // Iterate through each encoded token
        current_code += pair.second; // Append the Huffman code
        auto it = reverse_codes.find(current_code); // Search for the accumulated code in the reverse map
        if (it != reverse_codes.end()) { // If a matching code is found
            decompressed += it->second; // Append the corresponding character
            current_code.clear();       // Clear the current code for the next character
        }
    }
    return decompressed; // Return the decompressed string
}

// ----------------------- Run-Length Encoding Implementation -----------------------

/**
 * @brief Compresses a chunk of data using Run-Length Encoding (RLE).
 *
 * This function scans through the input data and creates RLE tokens by counting
 * consecutive occurrences of each character. Each token contains:
 * - character: The character being repeated.
 * - count: The number of consecutive repetitions of the character.
 *
 * @param data The input string to be compressed.
 * @return A vector of RLEToken representing the compressed data.
 */
std::vector<RLEToken> compress_rle(const std::string& data) {
    std::vector<RLEToken> tokens; // Vector to store the generated RLE tokens
    if (data.empty()) // Check if the input data is empty
        return tokens; // Return an empty vector if there's nothing to compress

    char current_char = data[0]; // Initialize the current character
    size_t count = 1;             // Initialize the count of consecutive characters

    for (size_t i = 1; i < data.size(); ++i) { // Iterate through the data starting from the second character
        if (data[i] == current_char) { // If the current character matches the previous one
            count++; // Increment the count
        }
        else { // If a different character is encountered
            tokens.emplace_back(RLEToken{ current_char, count }); // Add the token to the vector
            current_char = data[i]; // Update the current character
            count = 1;               // Reset the count
        }
    }
    tokens.emplace_back(RLEToken{ current_char, count }); // Add the final token
    return tokens; // Return the vector of generated tokens
}

/**
 * @brief Decompresses a chunk of RLE tokens to reconstruct the original data.
 *
 * This function iterates through the RLE tokens and reconstructs the original
 * data by repeating each character based on its count.
 *
 * @param tokens A vector of RLEToken representing the compressed data.
 * @return The decompressed original string.
 */
std::string decompress_rle(const std::vector<RLEToken>& tokens) {
    std::string decompressed; // String to store the decompressed data
    decompressed.reserve(tokens.size() * 2); // Reserve space to optimize memory allocations

    // Shared mutex for appending to decompressed string safely in parallel
    tbb::spin_mutex mutex;

    // Parallel loop to decompress the RLE tokens
    tbb::parallel_for(tbb::blocked_range<size_t>(0, tokens.size()),
        [&](const tbb::blocked_range<size_t>& range) {
            std::string local_decompressed; // Local string to store decompressed data in the thread
            for (size_t i = range.begin(); i < range.end(); ++i) { // Iterate through the assigned range
                local_decompressed.append(tokens[i].count, tokens[i].character); // Append the repeated characters
            }
            // Lock and append to the global decompressed string to avoid race conditions
            tbb::spin_mutex::scoped_lock lock(mutex);
            decompressed += local_decompressed; // Merge the local decompressed data
        });

    return decompressed; // Return the reconstructed original string
}

// ----------------------- Utility Functions Implementation -----------------------

// -----------------------------------------------------------------------------
/**
 * @brief Reads a specific chunk of a file.
 *
 * This function opens the specified file, seeks to the starting byte position,
 * and reads data up to the ending byte position. It handles cases where the
 * file cannot be opened or if the read operation fails.
 *
 * @param filename The name of the file to read from.
 * @param start The starting byte position in the file.
 * @param end The ending byte position in the file.
 * @return A string containing the data read from the specified chunk.
 */
std::string read_file_chunk(const std::string& filename, std::size_t start, std::size_t end) {
    std::ifstream file(filename, std::ios::binary); // Open file in binary mode
    if (!file.is_open()) { // Check if the file was successfully opened
        std::cerr << "Error opening file for reading: " << filename << "\n"; // Output error message
        return ""; // Return empty string if file cannot be opened
    }
    file.seekg(start); // Move to the start position
    std::size_t size = end - start; // Calculate the size of the chunk
    std::string buffer(size, '\0'); // Initialize buffer with null characters
    file.read(&buffer[0], size); // Read the chunk into the buffer
    file.close(); // Close the file after reading
    return buffer; // Return the read data
}

/**
 * @brief Template function to write compressed tokens to a file by appending.
 *
 * This function serializes the provided tokens and writes them to a binary file.
 * The file name is constructed using the base filename and the process rank to
 * ensure uniqueness across multiple MPI processes.
 *
 * @tparam T The type of tokens to write (e.g., LZ77Token, RLEToken).
 * @param filename The base name of the file to write to.
 * @param tokens A vector of tokens to be written.
 * @param rank The rank of the MPI process (used to differentiate file names).
 */
template <typename T>
void write_compressed_chunk(const std::string& filename, const std::vector<T>& tokens, int rank) {
    std::ofstream file;
    // Construct the file name with rank to avoid collisions
    file.open(filename + "_compressed_" + std::to_string(rank) + ".bin", std::ios::binary | std::ios::app);
    if (!file.is_open()) { // Check if the file was successfully opened
        std::cerr << "Error opening file for writing: " << filename << "\n"; // Output error message
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
 * This function reads binary data from a file corresponding to the given rank,
 * deserializes it into tokens, and returns them in a vector. It handles cases
 * where the file cannot be opened.
 *
 * @tparam T The type of tokens to read (e.g., LZ77Token, RLEToken).
 * @param filename The base name of the file to read from.
 * @param rank The rank of the MPI process (used to differentiate file names).
 * @return A vector of tokens read from the file.
 */
template <typename T>
std::vector<T> read_compressed_chunk(const std::string& filename, int rank) {
    std::vector<T> tokens; // Vector to store the read tokens
    // Construct the file name with rank to ensure correct file is read
    std::ifstream file(filename + "_compressed_" + std::to_string(rank) + ".bin", std::ios::binary);
    if (!file.is_open()) { // Check if the file was successfully opened
        std::cerr << "Error opening file for reading: " << filename << "\n"; // Output error message
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
 * This function opens the specified file in append mode and writes the
 * decompressed data to it. The file name is constructed using the base
 * filename and the process rank to ensure uniqueness across multiple MPI
 * processes.
 *
 * @param filename The base name of the file to write to.
 * @param data The decompressed data to be written.
 * @param rank The rank of the MPI process (used to differentiate file names).
 */
void write_decompressed_file(const std::string& filename, const std::string& data, int rank) {
    std::ofstream file;
    // Construct the decompressed file name with rank to avoid collisions
    file.open(filename + "_decompressed_" + std::to_string(rank) + ".txt", std::ios::app);
    if (!file.is_open()) { // Check if the file was successfully opened
        std::cerr << "Error opening file for writing: " << filename << "\n"; // Output error message
        return; // Exit the function if the file cannot be opened
    }
    file << data; // Write the decompressed data to the file
    file.close(); // Close the file after writing
}
