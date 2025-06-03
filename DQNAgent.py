import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 0.3  # Exploration rate
        self.epsilon_min = 0.01  # Minimum epsilon value
        self.epsilon_decay = 0.95  # Decay factor for epsilon
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))  # Output layer: Q-values for each action
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.action_size)
            print(f"Exploration: Random action {action}")
            return action
        else:
            q_values = self.model.predict(state)
            action = np.argmax(q_values[0])
            print(f"Exploitation: Q-values {q_values}, Action {action}")
            return action

    def remember(self, state, action, reward, next_state, done):
        """Stores experiences in memory."""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target += self.gamma * np.max(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        # Log epsilon updates
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        print(f"Epsilon: {self.epsilon}")

