import matplotlib as plt


# ADD FUNCTION THAT PLOTS

# Calculate the total counts for each mutation across patients
mutation_sums = data.sum(axis=1)

# Convert counts to frequencies
mutation_frequencies = mutation_sums / mutation_sums.sum()

# Group mutations into sets of 16 for coloring
num_groups = (len(mutation_frequencies) // 16) + 1
colors = plt.cm.tab20.colors  # Use a colormap with distinct colors

# Assign a color to each group of 16
bar_colors = [colors[i // 16 % len(colors)] for i in range(len(mutation_frequencies))]

# Plot the histogram
plt.figure(figsize=(14, 8))
bars = plt.bar(
    mutation_frequencies.index,
    mutation_frequencies.values,
    color=bar_colors,
    edgecolor="black",
)

# Customizing the x-axis for readability
plt.xticks(rotation=90, fontsize=8)
plt.title("Distribution of Mutation Frequencies (Grouped by 16)", fontsize=14)
plt.xlabel("Mutations", fontsize=12)
plt.ylabel("Frequency", fontsize=12)

# Add gridlines for better readability
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Add legend for groups of 16
handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(num_groups)]
labels = [f"Group {i+1}" for i in range(num_groups)]
plt.legend(handles, labels, title="Groups of 16 Mutations", bbox_to_anchor=(1.05, 1), loc="upper left")

# Show the plot
plt.tight_layout()
plt.show()
