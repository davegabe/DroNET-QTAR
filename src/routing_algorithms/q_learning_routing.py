from src.entities.uav_entities import Drone, HelloPacket, DataPacket
from src.routing_algorithms.BASE_routing import BASE_routing
from src.utilities import utilities as util
from typing import List, Tuple
import numpy as np


"""
Durante la fase di esplorazione usa il geo e poi piano piano impara.
La qTable è cellaDiPartenza x cellaDiArrivo
Migliora perchè non sceglie a caso in esplorazione ma perchè cerca di portare i pacchetti vicino al depot

"""


class QLearningRouting(BASE_routing):

    def __init__(self, drone, simulator):
        BASE_routing.__init__(self, drone, simulator)
        self.taken_actions = {}  # id event : (old_state, old_action)

        # set numpy seed
        np.random.seed(self.simulator.seed)

        # for each cell, i should have a list of the drones that are in that cell
        self.qTable = {}

        self.gamma = 0.5
        self.alpha = 0.1
        self.epsilon = 0.9

        # init the qTable for every cell and every drone
        x_cells = np.ceil(self.simulator.env_width /
                          self.simulator.prob_size_cell)
        y_cells = np.ceil(self.simulator.env_height /
                          self.simulator.prob_size_cell)
        self.n_cells = int(x_cells * y_cells)

        # our qTable will have shape cell x cell

        for i in range(self.n_cells):
            self.qTable[i] = [0] * self.n_cells

    def feedback(self, drone, id_event, delay, outcome):
        """
        Feedback returned when the packet arrives at the depot or
        Expire. This function have to be implemented in RL-based protocols ONLY
        @param drone: The drone that holds the packet
        @param id_event: The Event id
        @param delay: packet delay
        @param outcome: -1 or 1 (read below)
        @return:
        """

        if id_event in self.taken_actions:
            state, action = self.taken_actions[id_event]
            del self.taken_actions[id_event]

            partenza = int(state)
            arrivo = int(action[0])
            drone1 = action[1]

            if outcome == 1:
                reward = 2
            else:
                reward = -0.5

            # update the qTable
            self.qTable[partenza][arrivo] = self.qTable[partenza][arrivo] + self.alpha * (
                reward + self.gamma * max(self.qTable[arrivo]) - self.qTable[partenza][arrivo])

            # my nei

    def relay_selection(self, opt_neighbors: List[Tuple[HelloPacket, Drone]], packet: DataPacket) -> Drone:
        """
        This function returns the best relay to send packets.
        @param packet:
        @param opt_neighbors: a list of tuple (hello_packet, source_drone)
        @return: The best drone to use as relay
-       """

        my_cell_index = util.TraversedCells.coord_to_cell(size_cell=self.simulator.prob_size_cell,
                                                          width_area=self.simulator.env_width,
                                                          x_pos=self.drone.coords[0],
                                                          y_pos=self.drone.coords[1])[0]

        myDestinationCell = util.TraversedCells.coord_to_cell(size_cell=self.simulator.prob_size_cell,
                                                              width_area=self.simulator.env_width,
                                                              x_pos=self.drone.next_target()[0],
                                                              y_pos=self.drone.next_target()[1])[0]

        # state is a list of cells that a packet while the action is a list of drones
        state, action = None, None

        # to avoid problems with slices, we cast to int
        my_cell_index = int(my_cell_index)
        myDestinationCell = int(myDestinationCell)

        # dynamic epsilon

        if self.simulator.cur_step < 1000:
            self.epsilon = 0.9
        elif self.simulator.cur_step < 2000:
            self.epsilon = 0.7
        elif self.simulator.cur_step < 3000:
            self.epsilon = 0.3

        # epsilon case

        if np.random.uniform(0, 1) < self.epsilon:

            # choose the drone that is the closest to the depot. The closes is whoever has the euclidean distance min
            # between the drone and the depot
            minDistance = util.euclidean_distance(
                self.drone.coords, self.drone.depot.coords)

            for drone in opt_neighbors:
                destinationCell = util.TraversedCells.coord_to_cell(size_cell=self.simulator.prob_size_cell,
                                                                    width_area=self.simulator.env_width,
                                                                    x_pos=drone[1].next_target()[0],
                                                                    y_pos=drone[1].next_target()[1])[0]
                # euclidean distance between the drone and the depot
                drone = drone[1]
                distance = util.euclidean_distance(
                    drone.next_target(), self.drone.depot.coords)
                if distance < minDistance:
                    minDistance = distance
                    action = drone
                    state = destinationCell

            # once I've found the  min distance, I can look for the drones that have that value (i use a list because maybe two drones have the same distance)
            bestDrones = []
            for drone in opt_neighbors:
                drone = drone[1]
                distance = util.euclidean_distance(
                    drone.next_target(), self.drone.depot.coords)
                if distance == minDistance:
                    bestDrones.append(drone)
            if util.euclidean_distance(self.drone.coords, self.drone.depot.coords) == minDistance:
                bestDrones.append(self.drone)
            drone = np.random.choice(bestDrones)

            state = my_cell_index
            destinationCell = util.TraversedCells.coord_to_cell(size_cell=self.simulator.prob_size_cell,
                                                                width_area=self.simulator.env_width,
                                                                x_pos=drone.next_target()[0],
                                                                y_pos=drone.next_target()[1])[0]
            action = destinationCell

            state = int(state)
            action = (int(action), drone)

            self.taken_actions[packet.event_ref.identifier] = (state, action)
            return drone

        # explotaion scenario

        # the states are cells x cells
        # we can interpret them as source x destination
        max = self.qTable[my_cell_index][myDestinationCell]

        # find the best value
        for drone in opt_neighbors:
            drone = drone[1]

            destinationCell = util.TraversedCells.coord_to_cell(size_cell=self.simulator.prob_size_cell,
                                                                width_area=self.simulator.env_width,
                                                                x_pos=drone.next_target()[0],
                                                                y_pos=drone.next_target()[1])[0]
            destinationCell = int(destinationCell)
            if self.qTable[my_cell_index][destinationCell] > max:
                max = self.qTable[my_cell_index][destinationCell]

        # look for the drones with the best values
        bestDrones = []
        for drone in opt_neighbors:
            drone = drone[1]

            destinationCell = util.TraversedCells.coord_to_cell(size_cell=self.simulator.prob_size_cell,
                                                                width_area=self.simulator.env_width,
                                                                x_pos=drone.next_target()[0],
                                                                y_pos=drone.next_target()[1])[0]
            destinationCell = int(destinationCell)
            if self.qTable[my_cell_index][destinationCell] == max:
                bestDrones.append(drone)
        if self.qTable[my_cell_index][myDestinationCell] == max:
            bestDrones.append(self.drone)

        # choose randomly one of the best drones
        bestDrone = np.random.choice(bestDrones)

        state = my_cell_index
        action = (util.TraversedCells.coord_to_cell(size_cell=self.simulator.prob_size_cell,
                                                    width_area=self.simulator.env_width,
                                                    x_pos=bestDrone.next_target()[0],
                                                    y_pos=bestDrone.next_target()[1])[0], bestDrone)

        self.taken_actions[packet.event_ref.identifier] = (state, action)

        return bestDrone