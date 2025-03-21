import json
import random
from itertools import product

import modal
import numpy as np

# Define the Modal stub
app = modal.App("schelling-simulation")

# Define the Modal image with necessary dependencies
image = modal.Image.debian_slim().pip_install(["numpy"])


# Define the Modal function to run a simulation
@app.function(image=image, cpu=1, timeout=600)
def run_simulation(params):
    """Run a Schelling simulation with given parameters for fixed iterations"""
    empty_ratio = params["empty_ratio"]
    similarity_threshold = params["similarity_threshold"]
    size = params.get("size", 2500)
    n_neighbors = params.get("n_neighbors", 3)
    iterations = params.get("iterations", 15)
    seed = params.get("seed", 42)

    class Schelling:
        def __init__(
            self, size, empty_ratio, similarity_threshold, n_neighbors, seed=None
        ):
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

    # Create model and run simulation
    model = Schelling(size, empty_ratio, similarity_threshold, n_neighbors, seed)

    # Run for 15 iterations
    for _ in range(iterations):
        model.run()

    # Return result as a dictionary
    return {
        "empty_ratio": float(empty_ratio),
        "similarity_threshold": float(similarity_threshold),
        "mean_similarity": float(model.get_mean_similarity_ratio()),
    }


@app.local_entrypoint()
def main(output_file="schelling_results.json"):
    # Define parameter ranges with 0.01 step size
    empty_ratios = [round(x, 2) for x in np.arange(0.01, 1.00, 0.01)]
    similarity_thresholds = [round(x, 2) for x in np.arange(0.01, 1.00, 0.01)]

    # Create all parameter combinations
    param_combinations = list(product(empty_ratios, similarity_thresholds))
    params = [
        {"empty_ratio": er, "similarity_threshold": st} for er, st in param_combinations
    ]
    print(f"Running {len(params)} simulations with Modal...")

    # Run simulations in parallel
    results = list(run_simulation.map(params))

    # Save results to a file
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Completed {len(results)} simulations. Results saved to {output_file}")
    return results
