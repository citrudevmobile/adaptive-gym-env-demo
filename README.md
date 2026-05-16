# Adaptive Reinforcement Learning - King of the Hill

[![TensorFlow.js](https://img.shields.io/badge/TensorFlow.js-4.15.0-FF6F00?logo=tensorflow)](https://www.tensorflow.org/js)
[![Three.js](https://img.shields.io/badge/Three.js-r128-000000?logo=three.js)](https://threejs.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A complete educational implementation of **Adaptive Reinforcement Learning** using the King of the Hill game environment. This repository contains **two versions** of the same game, demonstrating the evolution from classical Q-learning to modern Deep Q-Networks (DQN) with TensorFlow.js.

## Table of Contents

- [What is Adaptive Reinforcement Learning?](#what-is-adaptive-reinforcement-learning)
- [The Game: King of the Hill](#the-game-king-of-the-hill)
- [Version Comparison](#version-comparison)
- [Version 1: Classical Q-Learning (Q-Table)](#version-1-classical-q-learning-q-table)
- [Version 2: Deep Q-Network (TensorFlow.js DQN)](#version-2-deep-q-network-tensorflowjs-dqn)
- [Quick Start](#quick-start)
- [File Structure](#file-structure)
- [How to Use](#how-to-use)
- [Understanding the Metrics](#understanding-the-metrics)
- [Educational Value](#educational-value)
- [Technical Details](#technical-details)
- [License](#license)

---

## What is Adaptive Reinforcement Learning?

**Normal Reinforcement Learning:**
- The environment rules never change
- The agent learns ONE strategy and uses it forever
- Example: A chess AI that always plays the same opening

**Adaptive Reinforcement Learning:**
- The environment rules CHANGE over time
- The agent must KEEP LEARNING and ADAPTING
- Example: A robot in a factory where machinery keeps getting moved around

In this King of the Hill game:
- The golden hill moves to random positions every N steps
- The speed N changes over time (faster, slower, random, or cyclic)
- The agent must continuously adapt to find the new hill location

---

## The Game: King of the Hill

**Rules:**
- You control a blue character in a 20x20 arena
- A golden hill appears somewhere in the arena
- You earn +1 point for every step you stand on the hill
- Every N steps, the hill teleports to a random new location
- N changes over time (this is the "adaptive" part!)
- Episode ends after 1000 steps

**Goal:** Maximize your score by staying on the hill as much as possible

---

## Version Comparison

| Feature | Version 1: Q-Table | Version 2: TensorFlow.js DQN |
|---------|-------------------|------------------------------|
| **File Name** | `adaptive-rl.js` | `tensorflow-adaptive-rl.js` |
| **Demo HTML** | `index.html` | `tensorflow-adaptive-rl-demo.html` |
| **Learning Method** | Tabular Q-Learning | Deep Q-Network (Neural Network) |
| **State Representation** | Discretized (8 bins per dimension) | Continuous (no discretization) |
| **Memory Structure** | Map (key-value pairs) | Experience Replay Buffer |
| **Number of Parameters** | Grows with states visited | Fixed (2,309 trainable parameters) |
| **Generalization** | None (only knows visited states) | Yes (can generalize to new states) |
| **Target Network** | Not used | Yes (for stable training) |
| **Loss Function** | N/A (direct update) | Mean Squared Error |
| **Optimizer** | N/A | Adam |
| **External Dependencies** | None | TensorFlow.js 4.15.0 |
| **Best For** | Teaching Q-learning fundamentals | Teaching modern deep RL |

---

## Version 1: Classical Q-Learning (Q-Table)

### How It Works

Version 1 uses a **Q-Table** - a big lookup table that stores a value for every possible (state, action) pair.

**State Discretization:**
- Continuous observations (like -0.32, 0.45) are put into bins
- 8 bins per dimension = 8^6 = 262,144 possible states
- Each state stores 5 Q-values (one per action)

**The Q-Learning Update (Bellman Equation):**
Q(s,a) = Q(s,a) + α × [ R + γ × max(Q(s',a')) - Q(s,a) ]


**Pros:**
- Simple to understand and implement
- No external libraries needed
- Works well for small state spaces
- Learning is transparent (you can inspect the Q-table)

**Cons:**
- Cannot handle continuous states directly (must discretize)
- Does not generalize to new states
- Memory grows with number of states encountered
- Not practical for complex problems (e.g., images, raw sensor data)

---

## Version 2: Deep Q-Network (TensorFlow.js DQN)

### How It Works

Version 2 uses a **Neural Network** to approximate Q-values instead of storing them in a table. The neural network learns to PREDICT the Q-value for any state, even ones it has never seen!

**Key DQN Concepts:**

| Concept | Explanation |
|---------|-------------|
| **Experience Replay** | Stores past experiences, learns from random batches. Prevents forgetting. |
| **Target Network** | A separate "slower" network used to calculate targets. Stabilizes training. |
| **Epsilon-Greedy** | Balances exploration (random actions) vs exploitation (learned actions). |
| **Bellman Equation** | Same as Q-learning, but applied to neural network outputs. |
| **Gradient Descent** | The algorithm that updates neural network weights to reduce prediction error. |

### Why Two Networks?

### Pros of Version 2

- **Handles continuous states** directly (no discretization needed)
- **Generalizes** - can make good guesses for never-before-seen states
- **Scales** to complex problems (images, audio, sensor data)
- **Fixed memory** - the neural network has a fixed number of parameters
- **State-of-the-art** - DQN is what powers modern RL successes (AlphaGo, Atari games)

### Cons of Version 2

- **More complex** - harder to understand the internal workings
- **Requires TensorFlow.js** - external library (~500KB)
- **Tuning required** - hyperparameters matter more
- **Less transparent** - can't easily inspect "what the network learned"

## The Adaptive Features (Both Versions)

Both versions share the same adaptive environment:

### Dynamic Goal Repositioning
![Game Screenshot](https://raw.githubusercontent.com/citrudevmobile/adaptive-gym-env-demo/Screenshot From 2026-05-16
