import matplotlib.pyplot as plt
from SumoEnvironment import SumoEnvironment
from DynamicFlowSumoEnvironment import DynamicFlowSumoEnvironment
from DQNAgent import DQNAgent
import numpy as np
import threading

env = DynamicFlowSumoEnvironment(sumo_cfg_file="ProjectFinal.sumocfg", traffic_light_id="J12")
agent = DQNAgent(state_size=env.state_size, action_size=env.action_size)

# Training parameters
episodes = 1
batch_size = 64
max_steps = 100  # Max steps per episode

# Track metrics across all episodes
episode_rewards = []
episode_lengths = []

# Replay threading
replay_thread = None

def replay_in_thread():
    """Run the replay function in a separate thread."""
    global replay_thread
    if replay_thread is None or not replay_thread.is_alive():
        replay_thread = threading.Thread(target=lambda: agent.replay(batch_size))
        replay_thread.start()

for episode in range(episodes):
    print(f"\nStarting Episode {episode + 1}/{episodes}")
    
    # Reset environment for the episode
    state = env.reset()
    state = np.reshape(state, [1, env.state_size])
    done = False
    total_reward = 0
    step_count = 0

    while not done:
        # Dynamically adjust traffic flow
        env.adjust_traffic_flow()

        # Get Q-values for the current state and log them
        q_values = agent.model.predict(state)
        print(f"Step {step_count + 1}, Q-values: {q_values}")

        # Select an action using the agent
        action = agent.act(state)
        
        # Perform the action and step the environment
        next_state, reward, done = env.step(action, step_count=step_count, max_steps=max_steps)
        next_state = np.reshape(next_state, [1, env.state_size])

        # Store experience in memory
        agent.remember(state, action, reward, next_state, done)

        # Update state and metrics
        state = next_state
        total_reward += reward
        step_count += 1

        # Trigger replay in a thread if memory size is sufficient
        if len(agent.memory) > batch_size:
            replay_in_thread()

    # Log episode metrics
    episode_rewards.append(total_reward)
    episode_lengths.append(step_count)
    print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}, Steps Taken: {step_count}")

# Wait for any ongoing replay to finish
if replay_thread is not None:
    replay_thread.join()

# After training, plot metrics
env.plot_metrics()

# Save the trained model
agent.model.save("results/dqn_model.h5")
print("Training complete. Model saved to results/dqn_model.h5.")

# Close the environment
env.close()
