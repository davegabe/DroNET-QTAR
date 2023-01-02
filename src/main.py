import numpy as np
from src.simulation.simulator import Simulator
from multiprocessing import Pool
import sys
import os
from src.utilities import config

OUTPUT_SUPPRESSED = True
MULTIPROCESS = True


def main():
    """ the place where to run simulations and experiments. """
    if MULTIPROCESS:
        avg_delivery_ratio = 0
        times = 10
        # Run multiprocess simulations
        with Pool(processes=times, initializer=mute) as pool:
            metrics = pool.map_async(run_sim, range(times)).get()
            delivery_ratios = [m[0] for m in metrics]
            mean_delivery_times = [m[1] for m in metrics]
            min_num_of_relays = [m[2] for m in metrics]
            avg_delivery_ratio = sum(delivery_ratios)/len(delivery_ratios)
            avg_mean_delivery_time = sum(mean_delivery_times)/len(mean_delivery_times)
        print("##########")
        print("Average delivery ratio: ", avg_delivery_ratio)
        print("Average mean delivery time: ", avg_mean_delivery_time)
        print("Average min number of relays: ", np.nanmean(min_num_of_relays))
        print("##########")
    else:
        # Run single process simulations
        run_sim(config.SEED)


def mute():
    if OUTPUT_SUPPRESSED:
        sys.stdout = open(os.devnull, 'w')


def run_sim(arg):
    """ Run a single simulation. """
    sim = Simulator(seed=arg)
    sim.run()            # run the simulation
    sim.metrics.other_metrics()  # calculate other metrics
    delivery_ratio = len(sim.metrics.drones_packets_to_depot_list) / sim.metrics.all_data_packets_in_simulation
    mean_delivery_time = sim.metrics.event_mean_delivery_time
    min_num_of_relays = np.nanmean(sim.metrics.mean_numbers_of_possible_relays)
    sim.close()
    return (delivery_ratio, mean_delivery_time, min_num_of_relays)


if __name__ == "__main__":
    main()