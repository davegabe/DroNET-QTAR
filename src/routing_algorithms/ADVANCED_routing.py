from src.entities.uav_entities import DataPacket, ACKPacket, Drone, HelloPacket, Packet
from typing import TYPE_CHECKING

from src.utilities import utilities as util
from src.utilities import config

from scipy.stats import norm
import abc

if TYPE_CHECKING:
    from src.simulation.simulator import Simulator


class ADVANCED_Routing(metaclass=abc.ABCMeta):

    def __init__(self, drone: Drone, simulator: 'Simulator'):
        """ The drone that is doing routing and simulator object. """

        self.drone = drone
        self.current_n_transmission = 0
        self.hello_messages = {}  # { drone_id : most recent hello packet}
        self.network_disp = simulator.network_dispatcher
        self.simulator = simulator

        if self.simulator.communication_error_type == config.ChannelError.GAUSSIAN:
            self.buckets_probability = self.__init_guassian()
        self.no_transmission = False

    @abc.abstractmethod
    def relay_selection(self, packet: DataPacket, drone_near_depot_id: int = -1) -> Drone:
        pass

    def routing_close(self):
        self.no_transmission = False

    def drone_reception(self, src_drone, packet: Packet, current_ts):
        """ handle reception an ACKs for a packets """
        if isinstance(packet, HelloPacket):
            src_id = packet.src_drone.identifier
            self.hello_messages[src_id] = packet  # add packet to our dictionary

        elif isinstance(packet, DataPacket):
            self.no_transmission = True
            self.drone.accept_packets([packet])
            # build ack for the reception
            ack_packet = ACKPacket(self.drone, src_drone, self.simulator, packet, current_ts)
            self.unicast_message(ack_packet, self.drone, src_drone, current_ts)

        elif isinstance(packet, ACKPacket):
            self.drone.remove_packets([packet.acked_packet])
            # packet.acked_packet.optional_data
            # print(self.is_packet_received_drone_reward, "ACK", self.drone.identifier)
            if self.drone.buffer_length() == 0:
                self.current_n_transmission = 0
                self.drone.move_routing = False

                # qui usare la gestione degli ack

    def drone_identification(self, drones: list[Drone], cur_step: int):
        """ handle drone hello messages to identify neighbors """
        # each drone sends a hello message to its neighbors every hello_interval time steps
        if cur_step % self.drone.hello_interval != 0:  # still not time to communicate
            return

        my_hello = HelloPacket(self.drone, cur_step, self.simulator, self.drone.coords,
                               self.drone.speed, self.drone.next_target(),
                               self.drone.link_holding_timer,
                               self.drone.one_hop_neighbors)

        self.broadcast_message(my_hello, self.drone, drones, cur_step)

    def routing(self, depot, drones, cur_step):
        # phase 1: Broadcast hello messages
        self.drone_identification(drones, cur_step)

        self.send_packets(cur_step)

        # close this routing pass
        self.routing_close()

    def send_packets(self, cur_step):
        """ procedure 3 -> choice next hop and try to send it the data packet """

        # FLOW 0
        if self.no_transmission or self.drone.buffer_length() == 0:
            return

        # FLOW 1
        if util.euclidean_distance(self.simulator.depot.coords, self.drone.coords) <= self.simulator.depot_com_range:
            # add error in case
            self.transfer_to_depot(self.drone.depot, cur_step)

            self.drone.move_routing = False
            self.current_n_transmission = 0
            return

        if cur_step % self.simulator.drone_retransmission_delta == 0:
            #################################
            ## ONE-HOP DISCOVERY ##
            # create a list of all the Drones that are in self.drone neighbourhood using hello messages
            opt_neighbors = []
            drone_near_depot_id = -1
            for hpk_id in self.hello_messages:
                hpk: HelloPacket = self.hello_messages[hpk_id]

                # check if packet is too old
                if hpk.time_step_creation < cur_step - config.OLD_HELLO_PACKET:
                    continue

                opt_neighbors.append((hpk, hpk.src_drone))

                # We have to check if the depot is in the communication range
                if util.euclidean_distance(self.simulator.depot.coords, hpk.src_drone.coords) <= self.simulator.depot_com_range:
                    # we cannot transmit to the depot since we don't have the packet yet
                    drone_near_depot_id = hpk.src_drone.identifier

            # update the list of one hop neighbors
            self.drone.prev_one_hop_neighbors = self.drone.one_hop_neighbors
            self.drone.one_hop_neighbors = [n[1] for n in opt_neighbors]

            # update hello interval
            self.drone.update_hello_interval(self.drone.one_hop_neighbors)

            #################################
            ## TWO-HOP DISCOVERY ##
            self.drone.two_hop_neighbors = dict()
            for hpk_id in self.hello_messages:
                hpk: HelloPacket = self.hello_messages[hpk_id]
                # two hop neighbors are the neighbors of the neighbor hpk_id
                two_hop_neighbors = hpk.one_hop_neighbors
                for n in two_hop_neighbors:
                    if n.identifier != self.drone.identifier:
                        if hpk_id not in self.drone.two_hop_neighbors:
                            self.drone.two_hop_neighbors[hpk_id] = [n]
                        else:
                            self.drone.two_hop_neighbors[hpk_id].append(n)

            # update hello interval
            two_hop_unique_neighbors: list[Drone] = []
            for drone_list in self.drone.two_hop_neighbors.values():
                for drone in drone_list:
                    if drone not in two_hop_unique_neighbors:
                        two_hop_unique_neighbors.append(drone)
            self.drone.update_hello_interval(two_hop_unique_neighbors)

            # if there are no neighbors, return
            if len(opt_neighbors) == 0:
                return

            # send packets
            for pkd in self.drone.all_packets():

                self.simulator.metrics.mean_numbers_of_possible_relays.append(len(opt_neighbors))

                best_neighbor = self.relay_selection(pkd, drone_near_depot_id)  # compute score

                if best_neighbor is not None:

                    self.unicast_message(pkd, self.drone, best_neighbor, cur_step)

                self.current_n_transmission += 1

    def geo_neighborhood(self, drones, no_error=False):
        """
        @param drones:
        @param no_error:
        @return: A list all the Drones that are in self.drone neighbourhood (no matter the distance to depot),
            in all direction in its transmission range, paired with their distance from self.drone
        """

        closest_drones = []  # list of this drone's neighbours and their distance from self.drone: (drone, distance)

        for other_drone in drones:

            if self.drone.identifier != other_drone.identifier:  # not the same drone
                drones_distance = util.euclidean_distance(self.drone.coords,
                                                          other_drone.coords)  # distance between two drones

                if drones_distance <= min(self.drone.communication_range,
                                          other_drone.communication_range):  # one feels the other & vv

                    # CHANNEL UNPREDICTABILITY
                    if self.channel_success(drones_distance, no_error=no_error):
                        closest_drones.append((other_drone, drones_distance))

        return closest_drones

    def channel_success(self, drones_distance, no_error=False):
        """
        Precondition: two drones are close enough to communicate. Return true if the communication
        goes through, false otherwise.
        """

        assert (drones_distance <= self.drone.communication_range)

        if no_error:
            return True

        if self.simulator.communication_error_type == config.ChannelError.NO_ERROR:
            return True

        elif self.simulator.communication_error_type == config.ChannelError.UNIFORM:
            return self.simulator.rnd_routing.rand() <= self.simulator.drone_communication_success

        elif self.simulator.communication_error_type == config.ChannelError.GAUSSIAN:
            return self.simulator.rnd_routing.rand() <= self.gaussian_success_handler(drones_distance)

    def broadcast_message(self, packet, src_drone, dst_drones, curr_step):
        """ send a message to my neigh drones"""
        for d_drone in dst_drones:
            self.unicast_message(packet, src_drone, d_drone, curr_step)

    def unicast_message(self, packet, src_drone, dst_drone, curr_step):
        """ send a message to my neigh drones"""
        # Broadcast using Network dispatcher
        self.simulator.network_dispatcher.send_packet_to_medium(packet, src_drone, dst_drone,
                                                                curr_step + config.LIL_DELTA)

    def gaussian_success_handler(self, drones_distance):
        """ get the probability of the drone bucket """
        bucket_id = int(drones_distance / self.radius_corona) * self.radius_corona
        return self.buckets_probability[bucket_id] * config.GUASSIAN_SCALE

    def transfer_to_depot(self, depot, cur_step):
        """ self.drone is close enough to depot and offloads its buffer to it, restarting the monitoring
                mission from where it left it
        """
        depot.transfer_notified_packets(self.drone, cur_step)
        self.drone.empty_buffer()
        self.drone.move_routing = False

    # --- PRIVATE ---
    def __init_guassian(self, mu=0, sigma_wrt_range=1.15, bucket_width_wrt_range=.5):

        # bucket width is 0.5 times the communication radius by default
        self.radius_corona = int(self.drone.communication_range * bucket_width_wrt_range)

        # sigma is 1.15 times the communication radius by default
        sigma = self.drone.communication_range * sigma_wrt_range

        max_prob = norm.cdf(mu + self.radius_corona, loc=mu, scale=sigma) - norm.cdf(0, loc=mu, scale=sigma)

        # maps a bucket starter to its probability of gaussian success
        buckets_probability = {}
        for bk in range(0, self.drone.communication_range, self.radius_corona):
            prob_leq = norm.cdf(bk, loc=mu, scale=sigma)
            prob_leq_plus = norm.cdf(bk + self.radius_corona, loc=mu, scale=sigma)
            prob = (prob_leq_plus - prob_leq) / max_prob
            buckets_probability[bk] = prob

        return buckets_probability
