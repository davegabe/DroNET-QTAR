import json
import matplotlib.pyplot as plt
import os
import time

def plot_avg_results(data: list[str], data_type: str, algorithms: list[str]):
    """Make a plot for a specific type of result (e.g. average_delivery_ratio, average_delivery_time, etc.) with all algorithms"""
    avg = {}
    found_algorithms = []
    for alg in algorithms:
        for file in data:
            if alg in file:
                found_algorithms.append(alg)
                with open(f"./results/{file}") as f:
                    file = json.load(f)
                # Each key is number of drones
                # Each value is a list of simulation results
                for key in file.keys():
                    avg[key] = sum(file[key]) / len(file[key])
                # Plot values
                plt.plot(list(avg.keys()), list(avg.values()))
                # Plot error bars
                # fmt small dots
                plt.errorbar(list(avg.keys()), list(avg.values()), yerr=0.1, capsize=2, fmt='.')
    # Set legend
    plt.legend(found_algorithms)

    # Set title and labels
    plt.title("All algorithms")
    plt.xlabel('Number of drones')
    plt.ylabel(data_type.replace('_', ' ').capitalize())
    # Save plot
    plt.savefig(f"./plots/{data_type}.png", dpi=400)
    # Close plot
    plt.close()

def main():
    # Create folders if they don't exist
    os.makedirs('./plots', exist_ok=True)
    os.makedirs('./results', exist_ok=True)
    # Get all json files from ./results folder
    json_files = [file for file in os.listdir('./results') if file.endswith('.json')]

    data_types = ['average_delivery_ratio', 'average_delivery_time', 'average_number_of_relays']
    algorithms = ["QTAR", "GEO", "RND", "QL"]

    # Plot values for each json file
    for data_type in data_types:
        # Get json files for each data type
        files_data_type = [file for file in json_files if data_type in file]

        plot_avg_results(files_data_type, data_type, algorithms)

if __name__ == '__main__':
    main()