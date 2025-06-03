import os
import sys
import traci
import numpy as np

class SumoEnvironment:
    def __init__(self, sumo_cfg_file, traffic_light_id):
        # Initialize SUMO environment settings
        self.sumo_cfg_file = sumo_cfg_file
        self.traffic_light_id = traffic_light_id
        self.lanes = ["E3_1", "E9_0", "E9_1"]  # Define lanes
        self.state_size = 3  # Set state size to 3 as per the get_state return size (traffic_density, waiting_time, queue_length)
        self.action_size = 5  # Actions: Green or Red light
        self._setup_sumo()

        self.cumulative_reward = 0  # Initialize cumulative reward

    def get_state(self):
        """
        Extract the current traffic state.
        :return: State as a NumPy array.
        """
        traffic_density = traci.lane.getLastStepVehicleNumber("E3_1") / traci.lane.getLength("E3_1")
        queue_length = traci.lane.getLastStepVehicleNumber("E9_0")
        waiting_time = traci.edge.getWaitingTime("E8")
        return np.array([traffic_density, waiting_time, queue_length])  # Return 3 values

    def reset(self):
        """Reset the environment."""
        try:
            # Try to start the SUMO connection (if already active, it will raise an error)
            traci.start(['sumo-gui', '-c', self.sumo_cfg_file])
        except traci.TraCIException:
            # If a connection is already active, we handle it by closing it and restarting
            print("Connection already active. Closing and restarting.")
            try:
                traci.close()  # Try to close the existing connection
            except traci.exceptions.FatalTraCIError:
                # If not connected, we just ignore the error
                pass
            traci.start(['sumo-gui', '-c', self.sumo_cfg_file])  # Start a new connection

        # Reset the state or return any other required value
        return self.get_state()

    def _setup_sumo(self):
        """Ensure SUMO_HOME is set and add its tools to PATH."""
        if "SUMO_HOME" not in os.environ:
            sys.exit("Please declare the environment variable 'SUMO_HOME'")
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)


    def take_action(self, action):
        """
        Apply an action to the traffic light.
        :param action: Action index corresponding to a traffic light configuration.
        """
        actions = [
            {"green_duration": 5, "red_duration": 25},
            {"green_duration": 10, "red_duration": 20},
            {"green_duration": 15, "red_duration": 15},
            {"green_duration": 20, "red_duration": 10},
            {"green_duration": 25, "red_duration": 5},
        ]
        selected_action = actions[action]
        traci.trafficlight.setPhaseDuration(self.traffic_light_id, selected_action["green_duration"])
        for _ in range(selected_action["red_duration"]):
            traci.simulationStep()

    def get_reward(self):
        """
        Compute the reward based on traffic metrics for multiple edges.
        :return: Reward value.
        """
        # Define the edges and their associated lanes for metric calculation
        edges = ["E5", "E3", "E6", "E8", "E9"]
        lanes = ["E9_0", "E5_0", "E6_0", "E8_0"]  # Example lanes to monitor queue lengths

        # Initialize metrics
        total_flow = 0
        total_waiting_time = 0
        total_queue_length = 0

        # Collect metrics for each edge
        for edge in edges:
            # Traffic flow: Number of vehicles passing through the edge
            total_flow += traci.edge.getLastStepVehicleNumber(edge)
            # Waiting time: Total waiting time on the edge
            total_waiting_time += traci.edge.getWaitingTime(edge)

        # Collect metrics for specific lanes
        for lane in lanes:
            # Queue length: Number of vehicles waiting in the lane
            total_queue_length += traci.lane.getLastStepVehicleNumber(lane)

        # Adjust weights to reflect the relative importance of different factors
        weights = {
            "flow_rate": 1.0,  # Encourage higher flow rates
            "waiting_time": -0.7,  # Discourage long waiting times
            "queue_length": -0.5  # Discourage long queues
        }

        # Calculate the weighted reward
        reward = (
            weights["flow_rate"] * total_flow +
            weights["waiting_time"] * total_waiting_time +
            weights["queue_length"] * total_queue_length
        )

        # Apply normalization or scaling to reward (optional)
        reward = reward / len(edges)  # Average over edges for consistency

        # Return the computed reward
        return reward



    def step(self, action, step_count, max_steps=100):
        """
        Perform an action, advance the simulation, and return the new state, reward, and done flag.
        :param action: The action to take (traffic light phase).
        :param step_count: Current step count for the simulation.
        :param max_steps: Maximum number of steps before ending the episode.
        :return: Next state, reward, and done flag.
        """
        # Apply the action (green or red light)
        self.take_action(action)
        
        # Step the simulation forward
        traci.simulationStep()

        # Get the current state after the simulation step
        state = self.get_state()

        # Calculate the reward for the current step
        reward = self.get_reward()

        # Accumulate the reward for future evaluations (cumulative rewards)
        self.cumulative_reward += reward

        # Set the 'done' flag based on the step count (not just when vehicles are left)
        done = step_count >= max_steps

        # Return the new state, cumulative reward, and 'done' flag
        return state, reward, done



    def close(self):
        """Close the SUMO simulation."""
        traci.close()
