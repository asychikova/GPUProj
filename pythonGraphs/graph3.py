import matplotlib.pyplot as plt

# Updated data from your table
processes = [2, 4, 6]
rle_times = [127102000, 143790000, 194937000]
lz77_times = [142965000000, 139791000000, 147357000000]
huffman_times = [746443974, 633230745, 999032040]

# Create the plot
plt.figure(figsize=(10, 6), facecolor='black')
ax = plt.gca()
ax.set_facecolor('black')

# Plot the data
plt.plot(processes, rle_times, label='RLE', marker='o', color='cyan')
plt.plot(processes, lz77_times, label='LZ77', marker='o', color='lime')
plt.plot(processes, huffman_times, label='Huffman', marker='o', color='yellow')

# Set labels and title
plt.xlabel('Number of Processes', color='white')
plt.ylabel('Compression Time (ns)', color='white')
plt.title('Compression Times for 4MB File with Different Algorithms', color='white')

# Set y-scale to logarithmic
plt.yscale('log')

# Customize tick labels
plt.xticks(processes, color='white')
plt.yticks(color='white')

# Add grid and legend
plt.grid(True, linestyle='--', alpha=0.6, color='white')
plt.legend(facecolor='black', edgecolor='white', labelcolor='white')

# Show the plot
plt.show()

