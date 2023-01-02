import sys
from src.utilities.experiments_config import *
from src.experiments.parser.parser import command_line_parser
from src.utilities import config
from src.simulation.simulator import Simulator
import os
from multiprocessing import Pool
from pprint import pprint
import numpy as np
import json
from datetime import datetime

OUTPUT_SUPPRESSED = True
MULTIPROCESS = True
ALG = "QTAR"
MAX_PROCESSES = 16


def sim_setup(n_drones, seed, algorithm):
    """
    Build an instance of Simulator using the parameters from src.utilities.experiments_config.py
    @param n_drones: the number of drones during the simulation
    @param seed: the simulation seed
    @param algorithm: the algorithm used to route the packets
    @return: an instance of Simulator
    """

    return Simulator(
        len_simulation=len_simulation,
        time_step_duration=time_step_duration,
        seed=seed,
        n_drones=n_drones,
        env_width=env_width,
        env_height=env_height,

        drone_com_range=drone_com_range,
        drone_sen_range=drone_sen_range,
        drone_speed=drone_speed,
        drone_max_buffer_size=drone_max_buffer_size,
        drone_max_energy=drone_max_energy,
        drone_retransmission_delta=drone_retransmission_delta,
        drone_communication_success=drone_communication_success,
        event_generation_delay=event_generation_delay,

        depot_com_range=depot_com_range,
        depot_coordinates=depot_coordinates,

        event_duration=event_duration,
        event_generation_prob=event_generation_prob,
        packets_max_ttl=packets_max_ttl,
        routing_algorithm=config.RoutingAlgorithm[algorithm],
        communication_error_type=config.ChannelError.GAUSSIAN,
        show_plot=show_plot,

        # ML parameters
        simulation_name="",

    )


def launch_experiments_oneseed(args):
    """
    The function launches simulations for a given algorithm and drones number
    with seeds ranging from in_seed up to out_seed
    @param n_drones: integer that describes the number of drones
    @param in_seed: integer that describe the initial seed
    @param out_seed: integer that describe the final seed
    @param algorithm: the routing algorithm
    @return:
    """
    n_drones, seed, algorithm = args
    print(f"Running {algorithm} with {n_drones} drones seed {seed}")
    sim = sim_setup(n_drones, seed, algorithm)
    sim.run()
    # Metrics
    sim.metrics.other_metrics()  # calculate other metrics
    delivery_ratio = len(sim.metrics.drones_packets_to_depot_list) / sim.metrics.all_data_packets_in_simulation
    mean_delivery_time = sim.metrics.event_mean_delivery_time
    min_num_of_relays = np.nanmean(sim.metrics.mean_numbers_of_possible_relays)
    sim.close()
    return delivery_ratio, mean_delivery_time, min_num_of_relays


def main():
    # Define the number of drones to test
    number_drones = [5, 10, 15, 20, 25, 30]
    # Define the number of seeds to test
    seeds = [i for i in range(30)]
    # For each number of drones, launch the simulation with the given seeds
    # save the results as a dictionary drone_number: [list of delivery ratios]
    delivery_ratios = {number_drone: [] for number_drone in number_drones}
    mean_delivery_times = {number_drone: [] for number_drone in number_drones}
    min_number_relays = {number_drone: [] for number_drone in number_drones}

    # Define the experiment parameters
    experiments = [(n_drones, seed, ALG) for n_drones in number_drones for seed in seeds]

    # Calculate experiments
    for i in range(0, len(experiments), MAX_PROCESSES):
        print("Starting batch", i, "to", i + min(MAX_PROCESSES, len(experiments) - i))
        with Pool(processes=MAX_PROCESSES, initializer=mute) as pool:
            metrics = pool.map_async(launch_experiments_oneseed, experiments[i:i+MAX_PROCESSES]).get()
            for j in range(len(metrics)):
                number_drone_result = experiments[i+j][0]
                # Extract the metrics
                delivery_ratio = metrics[j][0]
                mean_delivery_time = metrics[j][1]
                min_number_relay = metrics[j][2]
                # Save the results
                delivery_ratios[number_drone_result].append(delivery_ratio)
                mean_delivery_times[number_drone_result].append(mean_delivery_time)
                min_number_relays[number_drone_result].append(min_number_relay)

    # Print the results
    for number_drone in number_drones:
        print("#"*20)
        print(f"Number of drones: {number_drone}")
        print(f"Average delivery ratio: {sum(delivery_ratios[number_drone]) / len(delivery_ratios[number_drone])}")
        print(f"Average delivery time: {sum(mean_delivery_times[number_drone]) / len(mean_delivery_times[number_drone])}")
        print(f"Average number of relays: {sum(min_number_relays[number_drone]) / len(min_number_relays[number_drone])}")
        print("#"*20)
    if min(delivery_ratios) == 0:
        print("WARNING: some simulations failed")
    print("#"*20)
    print("Average (of averages) delivery ratio: ", sum([sum(delivery_ratios[number_drone]) / len(delivery_ratios[number_drone]) for number_drone in number_drones]) / len(number_drones))
    print("Average (of averages) delivery time: ", sum([sum(mean_delivery_times[number_drone]) / len(mean_delivery_times[number_drone]) for number_drone in number_drones]) / len(number_drones))
    print("Average (of averages) number of relays: ", sum([sum(min_number_relays[number_drone]) / len(min_number_relays[number_drone]) for number_drone in number_drones]) / len(number_drones))
    print("#"*20)
    # Save average delivery ratio
    with open(f"results/average_delivery_ratio_{ALG}.json", "w") as f:
        json.dump(delivery_ratios, f)
    # Save average delivery time
    with open(f"results/average_delivery_time_{ALG}.json", "w") as f:
        json.dump(mean_delivery_times, f)
    # Save average number of relays
    with open(f"results/average_number_of_relays_{ALG}.json", "w") as f:
        json.dump(min_number_relays, f)
        



def mute():
    if OUTPUT_SUPPRESSED:
        sys.stdout = open(os.devnull, 'w')


if __name__ == "__main__":
    main()