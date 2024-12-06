import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Provided data with no duplicates, sorted by input size and execution time
data = {
    "Algorithm": [
        "huffman", "lz77", "rle", "huffman", "lz77", "rle",
        "huffman", "lz77", "rle"
    ],
    "Input Size (Bytes)": [
        1000000, 1000000, 1000000, 2097152, 2097152, 2097152,
        4194304, 4194304, 4194304
    ],
    "Compression Time (ns)": [
        8.03E+09, 2.51E+09, 3.23E+07, 1.99E+10, 6.12E+09, 2.30E+08,
        4.01E+10, 1.10E+10, 1.25E+08
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Group and average the data to smooth out patterns
df_grouped = df.groupby(["Algorithm", "Input Size (Bytes)"], as_index=False)["Compression Time (ns)"].mean()

# Plot interpolated data for smoother curves
plt.figure(figsize=(12, 6))

# Define colors for each algorithm
colors = {"huffman": "blue", "lz77": "green", "rle": "red"}

# Plot each algorithm's performance
for algo in df_grouped["Algorithm"].unique():
    subset = df_grouped[df_grouped["Algorithm"] == algo]
    x = subset["Input Size (Bytes)"]
    y = subset["Compression Time (ns)"]
    x_new = np.logspace(np.log10(x.min()), np.log10(x.max()), 100)
    y_new = np.interp(x_new, x, y)  # Linear interpolation for smoothness

    plt.plot(x_new, y_new, label=algo.capitalize(), color=colors[algo])
    plt.scatter(x, y, color=colors[algo], marker='o')  # Original data points

# Add labels, title, and legend
plt.title("Compression Time by Algorithm and Increasing Input Size (Smoothed Curve)")
plt.xlabel("Input Size (Bytes)")
plt.ylabel("Compression Time (ns)")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()
