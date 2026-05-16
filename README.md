# Adaptive Reinforcement Learning - King of the Hill & Table Tennis

[![TensorFlow.js](https://img.shields.io/badge/TensorFlow.js-4.15.0-FF6F00?logo=tensorflow)](https://www.tensorflow.org/js)
[![Three.js](https://img.shields.io/badge/Three.js-r128-000000?logo=three.js)](https://threejs.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A complete educational implementation of **Adaptive Reinforcement Learning** featuring three progressive versions: from classical Q-learning to Deep Q-Networks (DQN) with TensorFlow.js, and finally a challenging Table Tennis environment with real physics.

## Table of Contents

- [What is Adaptive Reinforcement Learning?](#what-is-adaptive-reinforcement-learning)
- [The Games](#the-games)
- [Version Comparison](#version-comparison)
- [Version 1: Classical Q-Learning (Q-Table)](#version-1-classical-q-learning-q-table)
- [Version 2: Deep Q-Network (TensorFlow.js DQN)](#version-2-deep-q-network-tensorflowjs-dqn)
- [Version 3: Table Tennis (Physics + Prediction)](#version-3-table-tennis-physics--prediction)
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

---

## The Games

### Game 1: King of the Hill
![King of the Hill Screenshot](https://github.com/citrudevmobile/adaptive-gym-env-demo/blob/8b760148c77edb8b289189abae170451cb00e678/Screenshot%20From%202026-05-16%2017-21-26.png)

**Rules:**
- Control a blue character in a 20x20 arena
- A golden hill appears somewhere in the arena
- Earn +1 point for every step you stand on the hill
- Every N steps, the hill teleports to a random new location
- N changes over time (this is the "adaptive" part!)
- Episode ends after 1000 steps

**Goal:** Maximize your score by staying on the hill as much as possible

### Game 2: Table Tennis (Version 3)
A physics-based table tennis game where an AI opponent learns your playing patterns.

**Rules:**
- Control a paddle on a 3D table tennis court
- Hit the ball over the net before it bounces twice
- AI opponent uses neural network to track and return the ball
- Ball physics include gravity, bounce damping, and spin effects
- AI adapts to your playing style (learns where you hit)

**Why it's harder:**
- Ball has real physics (gravity, bounce, trajectory)
- Must PREDICT where the ball will land
- TIMING is critical (hit too early/late = miss)
- AI learns YOUR patterns over time

---

## Version Comparison

| Feature | Version 1: Q-Table | Version 2: TensorFlow.js DQN | Version 3: Table Tennis |
|---------|-------------------|------------------------------|------------------------|
| **Game** | King of the Hill | King of the Hill | Table Tennis |
| **File Name** | `adaptive-rl.js` | `tensorflow-adaptive-rl.js` | `table-tennis-adaptive-rl.js` |
| **Demo HTML** | `index.html` | `tensorflow-adaptive-rl-demo.html` | `table-tennis.html` |
| **Learning Method** | Tabular Q-Learning | Deep Q-Network (Neural Network) | Deep Q-Network |
| **State Space** | 6 dimensions | 6 dimensions | 14 dimensions |
| **Action Space** | 5 actions | 5 actions | 9 actions (move + shot types) |
| **Neural Network** | N/A | 6 → 64 → 64 → 5 | 14 → 128 → 128 → 64 → 9 |
| **Physics** | None | None | Gravity, bounce, damping, spin |
| **Prediction Required** | No | No | Yes (ball trajectory) |
| **Timing Critical** | No | No | Yes |
| **Memory Structure** | Map (key-value pairs) | Experience Replay Buffer | Experience Replay Buffer |
| **External Dependencies** | None | TensorFlow.js | TensorFlow.js |
| **Best For** | Q-learning basics | Modern deep RL | Complex physics + RL |

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
- Not practical for complex problems

---

## Version 2: Deep Q-Network (TensorFlow.js DQN)

### How It Works

Version 2 uses a **Neural Network** to approximate Q-values instead of storing them in a table. The neural network learns to PREDICT the Q-value for any state, even ones it has never seen!

**Neural Network Architecture:**
6 inputs → 64 neurons (ReLU) → 64 neurons (ReLU) → 5 outputs


**Key DQN Concepts:**

| Concept | Explanation |
|---------|-------------|
| **Experience Replay** | Stores past experiences, learns from random batches. Prevents forgetting. |
| **Target Network** | A separate "slower" network used to calculate targets. Stabilizes training. |
| **Epsilon-Greedy** | Balances exploration (random actions) vs exploitation (learned actions). |
| **Bellman Equation** | Same as Q-learning, but applied to neural network outputs. |
| **Gradient Descent** | Updates neural network weights to reduce prediction error. |

**Pros of Version 2:**
- Handles continuous states directly (no discretization needed)
- Generalizes to never-before-seen states
- Scales to complex problems (images, audio, sensor data)
- Fixed memory - neural network has fixed parameters

**Cons:**
- More complex - harder to understand internal workings
- Requires TensorFlow.js external library
- Hyperparameter tuning matters more

---

## Version 3: Table Tennis (Physics + Prediction)

### How It Works

Version 3 is a complete table tennis game with real physics and an AI opponent that learns from your playing style.

**Neural Network Architecture:**
14 inputs → 128 neurons (ReLU) → 128 neurons (ReLU) → 64 neurons (ReLU) → 9 outputs


**State Space (14 dimensions):**
- Ball position (x, y, z)
- Ball velocity (vx, vy, vz)
- AI paddle position (x, y)
- Opponent paddle position (x)
- Predicted landing position (x, z)
- Time to hit (frames)
- Last hit type
- Rally length

**Action Space (9 actions):**
- Move paddle (left, right, up, down)
- Hit types: flat, topspin, backspin, smash, lob

**Physics Simulation:**
- Gravity affects ball trajectory
- Bounce damping on table contact
- Spin affects ball curve
- Realistic paddle collision detection

**Adaptive Features:**
- AI learns your hitting patterns (left/right bias)
- Adapts strategy based on rally length
- Changes shot selection based on your weaknesses

---

## The Adaptive Features (All Versions)

All versions share adaptive environment features:

### Dynamic Goal Repositioning
The hill moves to random positions at random intervals (40-160 steps between moves).

### Adaptive Interval Changing
```javascript
// The hill move interval changes over time
strategies = ['random', 'faster', 'slower', 'cyclic'];
// Random: any number between min and max
// Faster: multiply by 0.7 (hill moves more often)
// Slower: multiply by 1.5 (hill moves less often)
// Cyclic: sine wave pattern
