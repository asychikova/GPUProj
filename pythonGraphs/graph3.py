import matplotlib.pyplot as plt

processes = [2, 4, 6]
rle_times = [201003172, 243340599, 273609075]
lz77_times = [261392000000, 891011000000, 1212410000000]
huffman_times = [5486696061, 6708920612, 4881003083]

plt.figure(figsize=(10, 6), facecolor='black') 
ax = plt.gca()  
ax.set_facecolor('black')  

plt.plot(processes, rle_times, label='RLE', marker='o', color='cyan')
plt.plot(processes, lz77_times, label='LZ77', marker='o', color='lime')
plt.plot(processes, huffman_times, label='Huffman', marker='o', color='yellow')

plt.xlabel('Number of Processes', color='white')
plt.ylabel('Compression Time (ns)', color='white')
plt.title('Compression Times for 4MB File with Different Algorithms', color='white')

plt.yscale('log')  
plt.xticks(processes, color='white')
plt.yticks(color='white')
plt.grid(True, linestyle='--', alpha=0.6, color='white')

plt.legend(facecolor='black', edgecolor='white', labelcolor='white')  

plt.show()