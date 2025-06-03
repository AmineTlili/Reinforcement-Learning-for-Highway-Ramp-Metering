import os
import sys
import traci
import numpy as np
import random
from SumoEnvironment import SumoEnvironment
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

class DynamicFlowSumoEnvironment(SumoEnvironment):
    def __init__(self, sumo_cfg_file, traffic_light_id):
        super().__init__(sumo_cfg_file, traffic_light_id)
        self.time_step = 0  # Track the simulation time step
        self.cumulative_reward = 0  # Initialize cumulative reward
        self.current_status = ""  # Store the current traffic status
        self.previous_status = ""  # Track the previous status
        self.traffic_status = []
        self.traffic_variation = [] 
        self.high_traffic_count = 0  # Count cars during high traffic
        self.low_traffic_count = 0  # Count cars during low traffic
        self.vehicle_counter = 0  # Unique vehicle ID counter
        self.actions_taken = []


        # Add lists to track metrics
        self.waiting_times = []  # Track total waiting time
        self.rewards = []  # Track rewards
        self.actions = []  # Track actions

    def adjust_traffic_flow(self):
        # Determine traffic condition
        if random.random() > 0.5:
            self.current_status = "High Traffic"
            if self.current_status != self.previous_status:
                if self.previous_status == "Low Traffic":
                    print(f"Low Traffic -> Total Cars Added: {self.low_traffic_count}")
                    self.low_traffic_count = 0
                elif self.previous_status == "High Traffic":
                    print(f"High Traffic -> Total Cars Added: {self.high_traffic_count}")
                    self.high_traffic_count = 0

            # Add cars and update counts for high traffic
            for lane_index in [0, 1]:
                for _ in range(5):
                    vehicle_id = f"dynamic_car_{self.vehicle_counter}"
                    self.vehicle_counter += 1
                    try:
                        traci.vehicle.add(vehicle_id, routeID="E5_to_E6_via_E3", depart=traci.simulation.getTime())
                        traci.vehicle.changeLane(vehicle_id, lane_index, duration=1000.0)
                        self.high_traffic_count += 1
                    except traci.TraCIException:
                        pass
        else:
            self.current_status = "Low Traffic"
            if self.current_status != self.previous_status:
                if self.previous_status == "High Traffic":
                    print(f"High Traffic -> Total Cars Added: {self.high_traffic_count}")
                    self.high_traffic_count = 0
                elif self.previous_status == "Low Traffic":
                    print(f"Low Traffic -> Total Cars Added: {self.low_traffic_count}")
                    self.low_traffic_count = 0

            # Add cars and update counts for low traffic
            lane_index = random.choice([0, 1])
            vehicle_id = f"dynamic_car_{self.vehicle_counter}"
            self.vehicle_counter += 1
            try:
                traci.vehicle.add(vehicle_id, routeID="E5_to_E6_via_E3", depart=traci.simulation.getTime())
                traci.vehicle.changeLane(vehicle_id, lane_index, duration=1000.0)
                self.low_traffic_count += 1
            except traci.TraCIException:
                pass

        # Update traffic status and variation
        self.traffic_status.append(self.current_status)
        self.traffic_variation.append({
            "High Traffic": self.high_traffic_count,
            "Low Traffic": self.low_traffic_count
        })
        self.previous_status = self.current_status


       


    def apply_action(self, action):
        """
        Apply the selected action to the traffic light.
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
        
        # Simulate for the red duration
        for _ in range(selected_action["red_duration"]):
            traci.simulationStep()

    def step(self, action, step_count, max_steps):
        """ Perform a step and track metrics. """
        self.time_step += 1  # Update the simulation step
        self.adjust_traffic_flow()  # Adjust traffic flow dynamically

        # Apply the action
        self.take_action(action)
        self.apply_action(action)

        # Run the simulation step
        traci.simulationStep()

        # Get the new state and reward
        state = self.get_state()
        reward = self.get_reward()

        # Update cumulative reward
        self.cumulative_reward += reward

        # Log metrics
        self.actions.append(action)
        self.rewards.append(reward)
        self.waiting_times.append(self.get_state()[1])  # Waiting time is the second element in the state
        self.actions_taken.append(action)

        # Log action and cumulative reward at each step
        print(f"Time Step: {self.time_step}, Action: {action}, Reward: {reward}, Cumulative Reward: {self.cumulative_reward}")

        # Set 'done' flag after a number of steps
        done = self.time_step >= max_steps

        return state, reward, done

    def plot_metrics(self):
        """Plot metrics after the simulation ends."""
        # Ensure that metrics lists are not empty
        if len(self.waiting_times) == 0 or len(self.rewards) == 0 or len(self.actions) == 0:
            print("Metrics data is empty, check data collection in the simulation.")
            return

        # Plot waiting times
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(self.waiting_times)), self.waiting_times, color='blue', alpha=0.7)
        plt.xlabel("Time Step")
        plt.ylabel("Waiting Time")
        plt.title("Waiting Time per Step")
        plt.savefig("results/plots/waiting_time.png")
        plt.grid()
        plt.show()

        # Plot rewards
        plt.figure(figsize=(12, 6))
        plt.plot(self.rewards, label="Reward", color='green')
        plt.xlabel("Time Step")
        plt.ylabel("Reward")
        plt.title("Reward Progression")
        plt.legend()
        plt.savefig("results/plots/rewards.png")
        plt.grid()
        plt.show()

        # Define the action durations (this can be placed in your class if it's not already there)
        action_durations = [
        {"green_duration": 5, "red_duration": 25},
        {"green_duration": 10, "red_duration": 20},
        {"green_duration": 15, "red_duration": 15},
        {"green_duration": 20, "red_duration": 10},
        {"green_duration": 25, "red_duration": 5},
        ]

        # Calculate frequency of each action
        action_counts = {i: self.actions.count(i) for i in range(len(action_durations))}

        # X-axis labels with action descriptions
        x_labels = [
            f"{i}: {action}" for i, action in enumerate(action_durations)
        ]

        # Plot action distribution
        plt.figure(figsize=(12, 6))
        plt.bar(action_counts.keys(), action_counts.values(), color='orange', alpha=0.7)
        plt.xticks(ticks=range(len(x_labels)), labels=x_labels, rotation=45, ha='right')
        plt.xlabel("Action (with Durations)")
        plt.ylabel("Frequency")
        plt.title("Action Distribution")
        plt.tight_layout()
        plt.savefig("results/plots/actions.png")
        plt.grid()
        plt.show()

        time_steps = range(len(self.traffic_variation))  # Ensure traffic_variation is updated each step
        total_vehicle_counts = [
            entry.get("High Traffic", 0) + entry.get("Low Traffic", 0) for entry in self.traffic_variation
        ]

        plt.figure(figsize=(12, 6))
        plt.plot(time_steps, total_vehicle_counts, label="Total Vehicles Passed", color='green')
        plt.xlabel("Time Step")
        plt.ylabel("Number of Vehicles")
        plt.title("Total Number of Vehicles Passed Over Time")
        plt.legend()
        plt.grid()
        plt.savefig("results/plots/vehicles_passed.png")
        plt.show()

        action_colors = ['blue', 'orange', 'green', 'red', 'purple']  # One color for each action (0, 1, 2, 3, 4)

        time_steps = range(len(self.actions_taken))  # Time steps correspond to the length of actions_taken

        plt.figure(figsize=(12, 6))

        # Use a bar plot with categorical y-axis
        plt.bar(
            time_steps,
            self.actions_taken,
            color=[action_colors[int(action)] for action in self.actions_taken],  # Ensure actions map to indices correctly
            alpha=0.8,
            edgecolor="black"
        )

        plt.xlabel("Time Step")
        plt.ylabel("Action")
        plt.title("Actions Taken Over Time")

        # Treat actions as categorical values on the y-axis
        plt.yticks(ticks=[0, 1, 2, 3, 4], labels=['Action 0', 'Action 1', 'Action 2', 'Action 3', 'Action 4'])

        plt.grid(axis='y', linestyle='--', alpha=0.7)


        legend_patches = [Patch(color=color, label=f"Action {i}") for i, color in enumerate(action_colors)]
        plt.legend(handles=legend_patches, title="Actions")
        # Save and show the plot
        plt.savefig("results/plots/actions_taken_bar.png")
        plt.show()
