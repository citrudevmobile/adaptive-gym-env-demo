# Adaptive RL Environment Demo

**Status:**  Working Code

A complete, visual demonstration of an **adaptive reinforcement learning environment** built in JavaScript/Three.js. The environment features a moving goal (King of the Hill) where the goal's movement interval **adapts over time**—testing an agent's ability to learn in non-stationary conditions.

Built to demonstrate rapid competency for the **Research Engineer in Adaptive RL Systems & Simulation Architecture** internship at Gearshift Fellowship.

---

## What This Demonstrates

| Feature | Implementation |
| :--- | :--- |
| **Dynamic Goal Repositioning** | The hill moves every N steps (adaptive interval: 40-160 steps) |
| **Environment Adaptation** | N changes over time (random, faster, slower, or cyclic patterns) |
| **Adaptive Reward Shaping** | Rewards change based on timing since goal moved |
| **Q-Learning Agent** | Tabular Q-learning with experience replay |
| **Adaptive Exploration** | Epsilon decays faster when performing well, slower when struggling |
| **Adaptive Learning Rate** | Adjusts based on performance trends |
| **Real-time 3D Visualization** | Built with Three.js |

---

## Files

| File | Purpose |
| :--- | :--- |
| `adaptive-rl.js` | Core adaptive RL module (Environment, Agent, Trainer classes) |
| `kingofthehill.html` | Complete 3D visualization with training UI |

---

## How to Run (Made Easy)

**Simply open `kingofthehill.html` in any modern web browser.**

No server required. No installation. No dependencies to install.

Just double-click the file and the demo runs immediately.

> Tested on Chrome, Firefox, and Edge

---

## How to Use the Demo

1. **Click "Train Agent"** – Watch the blue agent learn to chase the golden hill
2. **Observe adaptation** – The hill's movement interval changes over time (watch the "Goal Move Interval" metric)
3. **Watch metrics** – Sidebar shows exploration rate (ε), learning rate, rewards, and adaptation speed
4. **Stop/Reset** – Use buttons to pause or restart training

---

## Key Adaptive Features (For the Gearshift Fellowship)

### 1. Environment-Level Adaptation

The environment's *own behavior changes over time*:

```javascript
_adaptInterval() {
    // Strategies: random, faster, slower, cyclic
    // The goal can move more OR less frequently as training progresses
    this.currentInterval = newValue;
}