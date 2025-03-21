"""
title           : app.py
description     : Implementation of schelling segregation model using Python and Streamlit.
author          : Adil Moujahid
date_created    : 20200509
date_modified   : 20200509
version         : 0.1
usage           : streamlit run app.py
python_version  : 3.7.6
"""

import random

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.colors import ListedColormap


class Schelling:
    def __init__(self, size, empty_ratio, similarity_threshold, n_neighbors, seed=None):
        self.size = size
        self.empty_ratio = empty_ratio
        self.similarity_threshold = similarity_threshold
        self.n_neighbors = n_neighbors

        # Set random seeds for reproducibility
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Ratio of races (-1, 1) and empty houses (0)
        p = [(1 - empty_ratio) / 2, (1 - empty_ratio) / 2, empty_ratio]
        city_size = int(np.sqrt(self.size)) ** 2
        self.city = np.random.choice([-1, 1, 0], size=city_size, p=p)
        self.city = np.reshape(
            self.city, (int(np.sqrt(city_size)), int(np.sqrt(city_size)))
        )

    def get_neighborhood(self, row, col):
        """Get the neighborhood for a cell, implementing torus geometry"""
        rows, cols = self.city.shape
        neighborhood = []

        for i in range(-self.n_neighbors, self.n_neighbors + 1):
            for j in range(-self.n_neighbors, self.n_neighbors + 1):
                # Apply modulo to wrap around the grid
                neighbor_row = (row + i) % rows
                neighbor_col = (col + j) % cols
                neighborhood.append(self.city[neighbor_row, neighbor_col])

        return np.array(neighborhood)

    def run(self):
        rows, cols = self.city.shape

        # Get all non-empty positions (agents)
        non_empty_positions = list(zip(*np.where(self.city != 0)))

        # Shuffle positions to process agents in random order
        random.shuffle(non_empty_positions)

        # Iterate over agents in random order
        for row, col in non_empty_positions:
            race = self.city[row, col]
            neighborhood = self.get_neighborhood(row, col)
            # Remove the cell itself from the neighborhood count
            n_empty = np.count_nonzero(neighborhood == 0)
            n_similar = np.count_nonzero(neighborhood == race) - 1
            total_non_empty = len(neighborhood) - n_empty - 1

            # Skip if there are no other non-empty cells in the neighborhood
            if total_non_empty > 0:
                similarity_ratio = n_similar / total_non_empty
                # If below the threshold, move to a random empty house
                if similarity_ratio < self.similarity_threshold:
                    empty_positions = list(zip(*np.where(self.city == 0)))
                    if empty_positions:
                        new_pos = random.choice(empty_positions)
                        self.city[new_pos] = race
                        self.city[row, col] = 0

    def get_mean_similarity_ratio(self):
        total_ratio = 0
        count = 0
        rows, cols = self.city.shape
        for row in range(rows):
            for col in range(cols):
                race = self.city[row, col]
                if race != 0:
                    neighborhood = self.get_neighborhood(row, col)
                    n_empty = np.count_nonzero(neighborhood == 0)
                    n_similar = np.count_nonzero(neighborhood == race) - 1
                    total_non_empty = len(neighborhood) - n_empty - 1

                    if total_non_empty > 0:
                        similarity_ratio = n_similar / total_non_empty
                        total_ratio += similarity_ratio
                        count += 1

        return total_ratio / count if count > 0 else 0


# Streamlit App

st.title("Schelling's Model of Segregation")

population_size = st.sidebar.slider("Population Size", 500, 10000, 2500)
empty_ratio = st.sidebar.slider("Empty Houses Ratio", 0.0, 1.0, 0.2)
similarity_threshold = st.sidebar.slider("Similarity Threshold", 0.0, 1.0, 0.4)
n_iterations = st.sidebar.number_input("Number of Iterations", 10)
seed = st.sidebar.number_input("Random Seed (0 for random)", min_value=0, value=0)
if seed == 0:
    seed = None

schelling = Schelling(population_size, empty_ratio, similarity_threshold, 3, seed)
mean_similarity_ratio = []
mean_similarity_ratio.append(schelling.get_mean_similarity_ratio())

# Plot the graphs at initial stage
plt.style.use("ggplot")
plt.figure(figsize=(8, 4))

# Left hand side graph with Schelling simulation plot
cmap = ListedColormap(["red", "white", "royalblue"])
plt.subplot(121)
plt.axis("off")
plt.pcolor(schelling.city, cmap=cmap, edgecolors="w", linewidths=1)

# Right hand side graph with Mean Similarity Ratio graph
plt.subplot(122)
plt.xlabel("Iterations")
plt.xlim([0, n_iterations])
plt.ylim([0.4, 1])
plt.title("Mean Similarity Ratio", fontsize=15)
plt.text(
    1,
    0.95,
    "Similarity Ratio: %.4f" % schelling.get_mean_similarity_ratio(),
    fontsize=10,
)

city_plot = st.pyplot(plt)

progress_bar = st.progress(0)

if st.sidebar.button("Run Simulation"):
    for i in range(n_iterations):
        schelling.run()
        mean_similarity_ratio.append(schelling.get_mean_similarity_ratio())
        plt.figure(figsize=(8, 4))

        plt.subplot(121)
        plt.axis("off")
        plt.pcolor(schelling.city, cmap=cmap, edgecolors="w", linewidths=1)

        plt.subplot(122)
        plt.xlabel("Iterations")
        plt.xlim([0, n_iterations])
        plt.ylim([0.4, 1])
        plt.title("Mean Similarity Ratio", fontsize=15)
        plt.plot(range(1, len(mean_similarity_ratio) + 1), mean_similarity_ratio)
        plt.text(
            1,
            0.95,
            "Similarity Ratio: %.4f" % schelling.get_mean_similarity_ratio(),
            fontsize=10,
        )

        city_plot.pyplot(plt)
        plt.close("all")
        progress_bar.progress((i + 1.0) / n_iterations)
