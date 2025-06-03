# 🚦 Reinforcement Learning for Highway Ramp Metering using DQ
![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![SUMO](https://img.shields.io/badge/SUMO-Traffic%20Simulator-4B0082?logo=transportation)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Reinforcement Learning](https://img.shields.io/badge/Reinforcement%20Learning-DQN-green?logo=openai)
![Project](https://img.shields.io/badge/Academic-Project-lightgrey)

This project applies **Deep Q-Learning (DQN)** to optimize traffic light control for highway ramps. Built with **SUMO** (Simulation of Urban MObility) and Python, the system adapts in real-time to traffic conditions, minimizing vehicle wait times and maximizing flow efficiency.


This project applies **Deep Q-Learning (DQN)** to optimize traffic light control for highway ramps. Built with **SUMO** (Simulation of Urban MObility) and Python, the system adapts in real-time to traffic conditions, minimizing vehicle wait times and maximizing flow efficiency.

---

## 🧠 Problem Statement

Ramp metering is critical for reducing congestion and emissions on highways. Traditional methods use static rules. This project proposes a **reinforcement learning (RL)** approach to dynamically adjust signal timings at ramp entries.

---

## 🎯 Objectives

- Minimize average vehicle waiting time.
- Reduce queue lengths.
- Maximize vehicle throughput on highway ramps.
- Adapt to both high and low traffic conditions dynamically.

---

## 🛠️ Tools & Technologies

- 🐍 Python
- 🧠 TensorFlow / Keras
- 🚦 SUMO (Simulation of Urban Mobility)
- 🔁 TraCI (Traffic Control Interface)
- 📊 Matplotlib / Seaborn for visualizations

---

## 📐 Methodology

- **State Variables**: Traffic density, waiting time, and queue length per lane.
- **Action Space**: Different signal timing policies (e.g. 5s green/25s red, 15s/15s).
- **Reward Function**:
  R = w1 * R_queue + w2 * R_flow + w3 * R_waiting
- **RL Algorithm**: Deep Q-Network (DQN)
- Experience Replay
- Epsilon-Greedy Policy
- MSE loss optimization

---

## 🧪 Simulation Design

- Integrated SUMO and Python via TraCI API
- High and low traffic scenarios simulated dynamically
- Metrics Tracked:
- ✅ Average Waiting Time
- ✅ Queue Length
- ✅ Vehicle Throughput
- ✅ Cumulative Reward

---

## 📊 Results Overview

- ⬇️ Reduced waiting times and queue lengths
- ⬆️ Increased traffic throughput under dynamic flow
- 📈 Cumulative reward curve shows successful agent learning
- 📊 Visuals: Action distributions, traffic variations, and reward progression plots

---

## ⚠️ Challenges Faced

- RL complexity and hyperparameter tuning
- SUMO integration with TraCI and TensorFlow
- Realistic traffic simulation calibration


