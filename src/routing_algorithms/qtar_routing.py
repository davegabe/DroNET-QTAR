import numpy as np
from src.entities.uav_entities import DataPacket, Drone
from src.routing_algorithms.ADVANCED_routing import ADVANCED_Routing
from src.utilities import utilities as util


class QTARRouting(ADVANCED_Routing):

    def __init__(self, drone, simulator):
        ADVANCED_Routing.__init__(self, drone=drone, simulator=simulator)
        self.taken_actions = {}  # id event : (old_state, old_action)
        self.q_table = np.zeros(self.simulator.n_drones)
        self.A = 0.1  # delay importance
        self.B = 0.6  # speed importance
        self.C = 0.3  # energy importance
        self.rmin = -np.inf  # minimum reward
        self.rmax = -np.inf  # maximum reward

    def feedback(self, drone: Drone, id_event, delay, outcome: int = 0, is_destination: bool = False, is_local_minimum: bool = False):
        """
        Feedback returned when the packet arrives at the depot or
        Expire. This function have to be implemented in RL-based protocols ONLY
        @param drone: The drone that holds the packet
        @param id_event: The Event id
        @param delay: packet delay
        @param outcome: -1 if the packet expired; 1 if the packets has been delivered to the depot
        @return:
        """
        # Be aware, due to network errors we can give the same event to multiple drones and receive multiple
        # feedback for the same packet!!
        if id_event in self.taken_actions:
            state, action = self.taken_actions[id_event]

            # remove the entry, the action has received the feedback
            del self.taken_actions[id_event]

            # reward or update using the old state and the selected action at that time
            delay, speed, energy, norm_delay = state

            # learning rate
            lr = 1 - np.exp(-norm_delay) if norm_delay != -np.inf else 0.3
            
            # discount factor
            intersection_len = len(set(self.drone.one_hop_neighbors).intersection(set(self.drone.prev_one_hop_neighbors)))
            union_len = len(set(self.drone.one_hop_neighbors).union(set(self.drone.prev_one_hop_neighbors)))
            gamma = np.sqrt(intersection_len / union_len) if union_len != 0 else 0
            # reward
            reward = self.A * delay + self.B * speed + self.C * energy
            # set rmin and rmax
            self.rmin = min(self.rmin, reward) if self.rmin != -np.inf else reward
            self.rmax = max(self.rmax, reward) if self.rmax != -np.inf else reward
            # special cases for reward
            if is_destination:
                reward = self.rmax
            elif is_local_minimum:
                reward = self.rmin
            
            # update the q table
            self.q_table[action] = self.q_table[action] + lr * (reward + gamma * np.max(self.q_table) - self.q_table[action])

    def relay_selection(self, packet: DataPacket, drone_near_depot_id: int = -1) -> Drone:
        """
        This function returns the best relay to send packets.
        @param packet:
        @param opt_neighbors: a list of tuple (hello_packet, source_drone)
        @return: The best drone to use as relay
        """
        state: tuple[float, float, float, float] = (0,0,0,0)  # delay, speed, energy, norm_delay
        action: int = self.drone.identifier  # action is the drone id

        # potential good neighbors, ignore the ones that are too slow to reach the depot in time
        selected_drones: list[tuple[Drone, Drone]] = [] # (one_hop_neighbor, two_hop_neighbor)
        selected_sum_speed = 0
        selected_sum_energy = 0

        # otherwise select the neighbors that are faster than the required speed
        for one_hop_id, two_hop_neighbors in self.drone.two_hop_neighbors.items():
            one_hop_neighbor = self.simulator.drones[one_hop_id]
            for two_hop_neighbor in two_hop_neighbors:
                speed = util.two_hop_speed(self.drone, one_hop_neighbor, two_hop_neighbor, self.simulator)
                remaining_ttl = self.simulator.packets_max_ttl - (self.simulator.cur_step - packet.time_step_creation)
                required_speed = util.compute_required_speed(two_hop_neighbor, remaining_ttl, self.simulator)
                selected_sum_speed += speed
                selected_sum_energy += two_hop_neighbor.residual_energy / self.simulator.drone_max_energy
                if speed > required_speed:
                    selected_drones.append((one_hop_neighbor, two_hop_neighbor))

        # if there is a drone near the depot, select it
        if drone_near_depot_id != -1:
            selected_drone = self.simulator.drones[drone_near_depot_id]
            selected_drones = [ (selected_drone, selected_drone) ]
            selected_sum_speed += util.two_hop_speed(self.drone, selected_drone, self.simulator.depot, self.simulator)
            selected_sum_energy += selected_drone.residual_energy / self.simulator.drone_max_energy

        # select the best drone to relay the packet (the one with the highest q value) among the neighbors
        sorted_q_table = np.argsort(self.q_table)
        neighborsId = [d[1].identifier for d in selected_drones] # list of two hop neighbors ids
        for droneId in sorted_q_table:
            # check if the drone is in the list of neighbors by identifier
            if droneId in neighborsId:
                action = droneId
                # drone associated to the action (the id)
                action_drone = self.simulator.drones[action]
                # relay drone to reach the action drone
                one_hop_neighbor = selected_drones[neighborsId.index(action)][0]
                
                # compute the normalized delay
                norm_delay = [util.delay_between_drones(self.drone, one_hop_neighbor, self.simulator), util.delay_between_drones(one_hop_neighbor, action_drone, self.simulator)]
                norm_delay = np.abs(sum(norm_delay) - np.mean(norm_delay)) / np.var(norm_delay) if np.var(norm_delay) != 0 else -np.inf
                
                # compute the state
                delay = util.delay_between_drones(self.drone, action_drone, self.simulator)
                speed = util.two_hop_speed(self.drone, one_hop_neighbor, action_drone, self.simulator) / selected_sum_speed
                energy = action_drone.residual_energy / self.simulator.drone_max_energy / selected_sum_energy
                state = (delay, speed, energy, norm_delay)
                break

        # Store your current action --- you can add some stuff if needed to take a reward later
        self.taken_actions[packet.event_ref.identifier] = (state, action)

        is_destination = drone_near_depot_id != -1
        is_local_minimum = action == self.drone.identifier # and len(selected_drones) != 0
        self.feedback(self.drone, packet.event_ref.identifier, 0, 0, is_destination, is_local_minimum)

        return self.simulator.drones[action]
