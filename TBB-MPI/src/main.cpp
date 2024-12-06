#include "algorithms.hpp" // Ensure this includes declarations for read_file_chunk, compress_*, decompress_*, etc.
#include <mpi.h>           // MPI library for parallel processing
#include <iostream>        // For standard I/O operations
#include <fstream>         // For file I/O operations
#include <chrono>          // For high-resolution timing
#include <vector>          // For using std::vector
#include <string>          // For using std::string
#include <iomanip>         // For formatted I/O (e.g., std::setw)
#include <sstream>         // For string stream operations
#include <sys/stat.h>      // For file status information
#include <stdexcept>       // For exception handling

// -----------------------------------------------------------------------------
// Structure to hold performance metrics for each compression-decompression run
// -----------------------------------------------------------------------------
struct PerformanceMetrics {
    std::string algorithm;          // Compression algorithm used (e.g., huffman, lz77, rle)
    std::string filename;           // Name of the input file
    size_t input_size;              // Size of the input file in bytes
    size_t compressed_size;         // Size of the compressed data in bytes
    double compression_ratio;       // Ratio of compressed size to input size
    int num_processes;              // Number of MPI processes used

    // Time measurements (in nanoseconds) for different phases
    double compression_time;        // Time taken to compress the data
    double decompression_time;      // Time taken to decompress the data
    double communication_time;      // Time taken for MPI communications
    double total_time;              // Total time (compression + decompression + communication)

    // Performance indicators
    double speedup;                 // Speedup achieved compared to a single process
    double efficiency;              // Efficiency of parallelization (percentage)
    double throughput;              // Data processed per second (Bytes/s)
};

// -----------------------------------------------------------------------------
// Class to handle performance tracking by logging metrics to a CSV file
// -----------------------------------------------------------------------------
class PerformanceTracker {
private:
    std::ofstream csv_file_; // Output file stream for CSV logging

public:
    // Constructor: Initializes the CSV file and writes headers if the file is empty
    PerformanceTracker(const std::string& output_file) {
        csv_file_.open(output_file, std::ios::app); // Open file in append mode
        if (!csv_file_.is_open()) {
            throw std::runtime_error("Cannot open performance tracking file"); // Throw exception if file cannot be opened
        }

        // Write CSV headers if the file is empty
        csv_file_.seekp(0, std::ios::end); // Move to the end of the file
        if (csv_file_.tellp() == 0) {    // Check if the file is empty
            csv_file_ << "Algorithm,Filename,Input Size (Bytes),Compressed Size (Bytes),"
                << "Bytes Reduced,Compression Ratio,Processes,"
                << "Compression Time (ns),Decompression Time (ns),"
                << "Communication Time (ns),Total Time (ns),"
                << "Throughput (Bytes/s)\n";
        }
    }

    // Function to record and log performance metrics to the CSV file
    void recordPerformance(const PerformanceMetrics& metrics) {
        csv_file_ << metrics.algorithm << ","
            << metrics.filename << ","
            << metrics.input_size << ","
            << metrics.compressed_size << ","
            << (metrics.input_size - metrics.compressed_size) << ","
            << metrics.compression_ratio << ","
            << metrics.num_processes << ","
            << metrics.compression_time << ","
            << metrics.decompression_time << ","
            << metrics.communication_time << ","
            << metrics.total_time << ","
            << metrics.throughput << "\n";
        csv_file_.flush(); // Ensure data is written to the file immediately
    }

    // Destructor: Closes the CSV file if it's open
    ~PerformanceTracker() {
        if (csv_file_.is_open()) {
            csv_file_.close();
        }
    }
};

// -----------------------------------------------------------------------------
// Function to get the size of a file in bytes
// -----------------------------------------------------------------------------
std::size_t get_file_size(const std::string& filename) {
    struct stat stat_buf;                           // Structure to hold file status
    int rc = stat(filename.c_str(), &stat_buf);     // Get file status
    return rc == 0 ? stat_buf.st_size : 0;           // Return file size if successful, else 0
}

// -----------------------------------------------------------------------------
// Serialization functions to convert token vectors into binary strings
// -----------------------------------------------------------------------------

// Serialize Huffman tokens into a binary string
std::string serialize_huffman(const std::vector<std::pair<char, std::string>>& tokens) {
    std::ostringstream oss(std::ios::binary);        // Output string stream in binary mode
    for (const auto& token : tokens) {
        oss.write(&token.first, sizeof(char));        // Write the character
        size_t str_size = token.second.size();        // Get the size of the string
        oss.write(reinterpret_cast<const char*>(&str_size), sizeof(size_t)); // Write the string size
        oss.write(token.second.c_str(), str_size);    // Write the string data
    }
    return oss.str();                                 // Return the serialized binary string
}

// Serialize LZ77 tokens into a binary string
std::string serialize_lz77(const std::vector<LZ77Token>& tokens) {
    std::ostringstream oss(std::ios::binary);        // Output string stream in binary mode
    for (const auto& token : tokens) {
        oss.write(reinterpret_cast<const char*>(&token.offset), sizeof(int)); // Write the offset
        oss.write(reinterpret_cast<const char*>(&token.length), sizeof(int)); // Write the length
    }
    return oss.str();                                 // Return the serialized binary string
}

// Serialize RLE tokens into a binary string
std::string serialize_rle(const std::vector<RLEToken>& tokens) {
    std::ostringstream oss(std::ios::binary);        // Output string stream in binary mode
    for (const auto& token : tokens) {
        oss.write(&token.character, sizeof(char));     // Write the character
        oss.write(reinterpret_cast<const char*>(&token.count), sizeof(int)); // Write the count
    }
    return oss.str();                                 // Return the serialized binary string
}

// -----------------------------------------------------------------------------
// Deserialization functions to convert binary strings back into token vectors
// -----------------------------------------------------------------------------

// Deserialize binary string into Huffman tokens
std::vector<std::pair<char, std::string>> deserialize_huffman(const std::string& serialized_data) {
    std::vector<std::pair<char, std::string>> tokens; // Vector to hold deserialized tokens
    std::istringstream iss(serialized_data, std::ios::binary); // Input string stream in binary mode
    while (iss) {
        char character;
        size_t str_size;
        iss.read(&character, sizeof(char));            // Read the character
        if (iss.gcount() < sizeof(char)) break;        // Break if not enough data
        iss.read(reinterpret_cast<char*>(&str_size), sizeof(size_t)); // Read the string size
        if (iss.gcount() < sizeof(size_t)) break;      // Break if not enough data
        std::string str(str_size, '\0');               // Initialize string with null characters
        iss.read(&str[0], str_size);                   // Read the string data
        if (iss.gcount() < static_cast<std::streamsize>(str_size)) break; // Break if not enough data
        tokens.emplace_back(character, str);           // Add the token to the vector
    }
    return tokens;                                     // Return the deserialized tokens
}

// Deserialize binary string into LZ77 tokens
std::vector<LZ77Token> deserialize_lz77(const std::string& serialized_data) {
    std::vector<LZ77Token> tokens;                     // Vector to hold deserialized tokens
    std::istringstream iss(serialized_data, std::ios::binary); // Input string stream in binary mode
    while (iss) {
        LZ77Token token;
        iss.read(reinterpret_cast<char*>(&token.offset), sizeof(int)); // Read the offset
        if (iss.gcount() < sizeof(int)) break;        // Break if not enough data
        iss.read(reinterpret_cast<char*>(&token.length), sizeof(int)); // Read the length
        if (iss.gcount() < sizeof(int)) break;        // Break if not enough data
        tokens.push_back(token);                       // Add the token to the vector
    }
    return tokens;                                     // Return the deserialized tokens
}

// Deserialize binary string into RLE tokens
std::vector<RLEToken> deserialize_rle(const std::string& serialized_data) {
    std::vector<RLEToken> tokens;                      // Vector to hold deserialized tokens
    std::istringstream iss(serialized_data, std::ios::binary); // Input string stream in binary mode
    while (iss) {
        RLEToken token;
        iss.read(&token.character, sizeof(char));       // Read the character
        if (iss.gcount() < sizeof(char)) break;        // Break if not enough data
        iss.read(reinterpret_cast<char*>(&token.count), sizeof(int)); // Read the count
        if (iss.gcount() < sizeof(int)) break;          // Break if not enough data
        tokens.push_back(token);                        // Add the token to the vector
    }
    return tokens;                                      // Return the deserialized tokens
}

// -----------------------------------------------------------------------------
// Main function: Orchestrates the compression-decompression pipeline and performance tracking
// -----------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv); // Initialize the MPI environment

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);   // Get the rank of the current process
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);   // Get the total number of processes

    // Initialize PerformanceTracker only on the root process (rank 0)
    PerformanceTracker* tracker = nullptr;
    if (world_rank == 0) {
        try {
            tracker = new PerformanceTracker("compression_performance.csv"); // Create a PerformanceTracker instance
        }
        catch (const std::exception& e) { // Catch any exceptions during initialization
            std::cerr << e.what() << "\n"; // Output the error message
            MPI_Abort(MPI_COMM_WORLD, 1);  // Abort the MPI program with an error code
        }
    }

    // List of input files to be processed
    std::vector<std::string> input_files = {
        "input.txt",
        "input1MB.txt",
        "input2MB.txt",
        "input4MB.txt"
    };

    // List of compression algorithms to be applied
    std::vector<std::string> algorithms = {
        "huffman",
        "lz77",
        "rle"
    };

    // Iterate over each input file
    for (const auto& file : input_files) {
        std::size_t file_size = 0; // Variable to store the size of the current file

        // Only the root process reads the file size
        if (world_rank == 0) {
            file_size = get_file_size(file); // Get the size of the file
            if (file_size == 0) {            // Check if the file is empty or cannot be read
                std::cerr << "Input file " << file << " is empty or cannot be read.\n";
            }
        }

        // Broadcast the file size to all processes
        MPI_Bcast(&file_size, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

        if (file_size == 0) {
            continue; // Skip to the next file if the current file is empty or unreadable
        }

        // Calculate the chunk size for each process
        std::size_t chunk_size = file_size / world_size;                  // Size of data chunk per process
        std::size_t start = world_rank * chunk_size;                      // Start position for the current process
        std::size_t end = (world_rank == world_size - 1) ? file_size : start + chunk_size; // End position

        // Read the assigned chunk of data from the file
        std::string data = read_file_chunk(file, start, end);
        if (data.empty()) { // Check if data reading was successful
            std::cerr << "Process " << world_rank << " failed to read data from " << file << ".\n";
            continue; // Skip to the next algorithm if reading failed
        }

        // Iterate over each compression algorithm
        for (const auto& algorithm : algorithms) {
            PerformanceMetrics metrics;      // Instantiate a PerformanceMetrics object to store metrics
            metrics.algorithm = algorithm;   // Set the algorithm name
            metrics.filename = file;         // Set the filename
            metrics.input_size = file_size;  // Set the input file size
            metrics.num_processes = world_size; // Set the number of MPI processes

            // ------------------------------------------
            // Compression Phase
            // ------------------------------------------

            // Start the compression timer
            auto compression_start = std::chrono::high_resolution_clock::now();

            std::string compressed_data; // Variable to hold serialized compressed data

            // Perform compression based on the selected algorithm
            if (algorithm == "huffman") {
                auto compressed_tokens = compress_huffman(data);             // Compress data using Huffman
                compressed_data = serialize_huffman(compressed_tokens);      // Serialize the compressed tokens
            }
            else if (algorithm == "lz77") {
                auto compressed_tokens = compress_lz77(data);                // Compress data using LZ77
                compressed_data = serialize_lz77(compressed_tokens);         // Serialize the compressed tokens
            }
            else if (algorithm == "rle") {
                auto compressed_tokens = compress_rle(data);                 // Compress data using RLE
                compressed_data = serialize_rle(compressed_tokens);          // Serialize the compressed tokens
            }
            else {
                // Handle unknown algorithms gracefully
                if (world_rank == 0) {
                    std::cerr << "Unknown algorithm: " << algorithm << "\n";
                }
                continue; // Skip to the next algorithm if the current one is unknown
            }

            // Define the name of the compressed output file
            std::string compressed_filename = "compressed_" + algorithm + "_" + std::to_string(world_rank) + ".bin";

            // Open the compressed file in binary write mode
            std::ofstream ofs(compressed_filename, std::ios::binary);
            if (!ofs) { // Check if the file was opened successfully
                std::cerr << "Failed to open " << compressed_filename << " for writing.\n";
                continue; // Skip to the next algorithm if file writing failed
            }

            // Write the serialized compressed data to the file
            ofs.write(compressed_data.c_str(), compressed_data.size());
            ofs.close(); // Close the file after writing

            metrics.compressed_size = compressed_data.size(); // Record the size of the compressed data

            // Stop the compression timer and calculate the elapsed time
            auto compression_end = std::chrono::high_resolution_clock::now();
            metrics.compression_time = std::chrono::duration<double, std::nano>(compression_end - compression_start).count();

            // ------------------------------------------
            // Communication Phase (MPI Gather)
            // ------------------------------------------

            // Start the communication timer
            auto communication_start = std::chrono::high_resolution_clock::now();

            size_t local_compressed_size = metrics.compressed_size; // Size of compressed data from the current process
            std::vector<size_t> all_compressed_sizes;               // Vector to hold compressed sizes from all processes

            if (world_rank == 0) {
                all_compressed_sizes.resize(world_size);            // Resize the vector on the root process
            }

            // Gather compressed sizes from all processes to the root process
            MPI_Gather(&local_compressed_size, 1, MPI_UNSIGNED_LONG,    // Send buffer: address of local_compressed_size
                all_compressed_sizes.data(), 1, MPI_UNSIGNED_LONG, // Receive buffer: all_compressed_sizes
                0, MPI_COMM_WORLD);                                 // Root process and communicator

            // Stop the communication timer and calculate the elapsed time
            auto communication_end = std::chrono::high_resolution_clock::now();
            metrics.communication_time = std::chrono::duration<double, std::nano>(communication_end - communication_start).count();

            // ------------------------------------------
            // Compression Ratio Calculation
            // ------------------------------------------

            double compression_ratio = 0.0; // Initialize compression ratio
            size_t total_compressed_size = 0; // Initialize total compressed size

            if (world_rank == 0) { // Only the root process calculates the compression ratio
                for (const auto& size : all_compressed_sizes) {
                    total_compressed_size += size; // Sum up compressed sizes from all processes
                }
                compression_ratio = (file_size > 0) ? static_cast<double>(total_compressed_size) / static_cast<double>(file_size) : 0.0; // Calculate ratio
                metrics.compression_ratio = compression_ratio; // Record the compression ratio
            }

            // Broadcast the compression ratio from the root process to all other processes
            MPI_Bcast(&compression_ratio, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            // ------------------------------------------
            // Total Time Calculation
            // ------------------------------------------

            metrics.total_time = metrics.compression_time + metrics.communication_time; // Total time includes compression and communication

            // ------------------------------------------
            // Throughput Calculation
            // ------------------------------------------

            double throughput = 0.0; // Initialize throughput

            if (world_rank == 0) { // Only the root process calculates throughput
                double total_time_seconds = metrics.total_time / 1e9; // Convert nanoseconds to seconds
                throughput = (total_time_seconds > 0) ? static_cast<double>(file_size) / total_time_seconds : 0.0; // Calculate throughput
                metrics.throughput = throughput; // Record throughput
            }

            // Broadcast throughput from the root process to all other processes (optional)
            MPI_Bcast(&throughput, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            // ------------------------------------------
            // Record Performance Metrics
            // ------------------------------------------

            if (world_rank == 0 && tracker != nullptr) { // Only the root process records metrics
                tracker->recordPerformance(metrics);      // Log metrics to the CSV file
            }

            // ------------------------------------------
            // Console Output (Optional)
            // ------------------------------------------

            if (world_rank == 0) { // Only the root process outputs to the console
                std::cout << std::left << std::setw(12) << algorithm       // Algorithm name
                    << std::setw(15) << file                // Filename
                    << std::setw(20) << file_size          // Input file size
                    << std::setw(25) << total_compressed_size // Total compressed size
                    << std::setw(18) << (file_size - total_compressed_size) // Bytes reduced
                    << std::setw(20) << compression_ratio   // Compression ratio
                    << std::setw(12) << world_size         // Number of processes
                    << std::setw(25) << metrics.compression_time    // Compression time
                    << std::setw(25) << metrics.decompression_time  // Decompression time (not measured here)
                    << std::setw(20) << metrics.communication_time  // Communication time
                    << std::setw(15) << metrics.total_time          // Total time
                    << std::setw(20) << throughput                 // Throughput
                    << "\n";
            }

            // ------------------------------------------
            // Synchronize All Processes Before Next Algorithm
            // ------------------------------------------
            MPI_Barrier(MPI_COMM_WORLD); // Ensure all processes reach this point before proceeding
        }
    }

    // ------------------------------------------
    // Cleanup: Delete the PerformanceTracker instance on the root process
    // ------------------------------------------
    if (world_rank == 0 && tracker != nullptr) {
        delete tracker; // Free the allocated PerformanceTracker memory
    }

    // ------------------------------------------
    // Finalize the MPI Environment and Exit
    // ------------------------------------------
    MPI_Finalize(); // Terminate the MPI environment
    return 0;
}
