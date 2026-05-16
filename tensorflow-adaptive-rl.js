/**
 * ============================================================================
 * TENSORFLOW.JS ADAPTIVE REINFORCEMENT LEARNING MODULE
 * ============================================================================
 * 
 * A complete Deep Q-Network (DQN) implementation for the King of the Hill game.
 * This module contains the AI agent that learns to play using a neural network.
 * 
 * ============================================================================
 * WHAT IS REINFORCEMENT LEARNING?
 * ============================================================================
 * 
 * Reinforcement Learning is a type of machine learning where an AGENT learns
 * by interacting with an ENVIRONMENT. The agent takes ACTIONS, receives
 * REWARDS (positive or negative), and learns which actions are good or bad.
 * 
 * Think of it like training a dog:
 * - Dog does something good → Give treat (POSITIVE REWARD)
 * - Dog does something bad → No treat (NEGATIVE REWARD)
 * - Dog learns to repeat good behaviors
 * 
 * In our game:
 * - AGENT = The red AI robot
 * - ENVIRONMENT = The game arena with the hill
 * - ACTION = Move up, down, left, right, or stay
 * - REWARD = +1 point for standing on the hill, -0.01 for time waster
 * 
 * ============================================================================
 * WHAT IS Q-LEARNING?
 * ============================================================================
 * 
 * Q-Learning is a specific RL algorithm that learns the VALUE of each action
 * in each state. This value is called the Q-VALUE.
 * 
 * Q(s, a) = How good is it to take action 'a' when in state 's'?
 * 
 * Example:
 * - State: Agent at (5, 3), Hill at (0, 0), close to hill
 * - Q(Up) = 0.8    (moving up is good - gets closer)
 * - Q(Down) = -0.5 (moving down is bad - moves away)
 * - Q(Left) = 0.3  (moving left is okay)
 * - Q(Right) = 0.1 (moving right is not great)
 * - Q(Stay) = -0.2 (staying is bad - wasting time)
 * 
 * The agent always tries to take the action with the highest Q-value.
 * 
 * ============================================================================
 * WHAT IS A NEURAL NETWORK?
 * ============================================================================
 * 
 * A neural network is a computer program inspired by the human brain.
 * It consists of layers of "neurons" that process information.
 * 
 * Our Neural Network Architecture:
 * 
 *     INPUT LAYER           HIDDEN LAYER 1      HIDDEN LAYER 2        OUTPUT LAYER
 *    (6 neurons)             (64 neurons)         (64 neurons)         (5 neurons)
 *         │                       │                    │                   │
 *    ┌────┼────┐             ┌─────┼─────┐          ┌─────┼─────┐        ┌─────┼─────┐
 *    │    │    │             │     │     │          │     │     │        │     │     │
 *    ▼    ▼    ▼             ▼     ▼     ▼          ▼     ▼     ▼        ▼     ▼     ▼
 *   [●]  [●]  [●]    →     [●●●]  [●●●]  [●●●]  →  [●●●]  [●●●]  [●●●]  → [●]  [●]  [●]
 *    │    │    │             │     │     │          │     │     │        │     │     │
 *    └────┼────┘             └─────┼─────┘          └─────┼─────┘        └─────┼─────┘
 *         │                       │                    │                   │
 *    Agent X, Agent Z         Learns patterns       Learns complex       Q-Value for:
 *    Hill X, Hill Z           like "if far          patterns like        0 = Move Up
 *    Time until move          from hill,            "if opponent         1 = Move Down
 *    In zone flag             move toward it"       is winning,          2 = Move Left
 *                                                    play aggressive"     3 = Move Right
 *                                                                         4 = Stay
 * 
 * WHAT IS ReLU ACTIVATION?
 * ------------------------
 * ReLU = Rectified Linear Unit. Formula: f(x) = max(0, x)
 * - If input is negative → output 0 (neuron "turns off")
 * - If input is positive → output same number (neuron "fires")
 * This helps the network learn non-linear patterns.
 * 
 * ============================================================================
 * THE BELLMAN EQUATION (How Q-Learning Works)
 * ============================================================================
 * 
 * The Bellman equation is the formula that updates Q-values:
 * 
 *     Q(s,a) = Q(s,a) + α × [ R + γ × max(Q(s',a')) - Q(s,a) ]
 * 
 * Let's break this down with an example:
 * 
 * Imagine you're playing a video game:
 * - You are at level 1 (state s)
 * - You decide to jump over a pit (action a)
 * - You succeed and get 100 points (reward R)
 * - Now you're at level 2 (next state s')
 * - In level 2, the best action is worth 200 points (max Q(s',a'))
 * 
 * The equation calculates: 
 *   New knowledge = R + γ × best future = 100 + 0.95 × 200 = 290
 * 
 * Then it updates your old estimate:
 *   New Q = Old Q + α × (New knowledge - Old Q)
 *   New Q = 50 + 0.1 × (290 - 50) = 74
 * 
 * Your estimate for jumping went from 50 to 74!
 * 
 * PARAMETERS EXPLAINED:
 * --------------------
 * α (alpha) = Learning Rate (0.001)
 *   - How much to trust new information
 *   - High = learns fast but might forget
 *   - Low = learns slow but stable
 * 
 * γ (gamma) = Discount Factor (0.95)
 *   - How much to value future rewards
 *   - 1.0 = future is as important as now
 *   - 0.5 = only care about immediate rewards
 * 
 * ============================================================================
 * DEEP Q-NETWORK (DQN) INNOVATIONS
 * ============================================================================
 * 
 * Normal Q-learning uses a Q-table (big spreadsheet) to store values.
 * Problem: What if there are millions of possible states? Table gets too big!
 * 
 * DQN Solution: Use a NEURAL NETWORK to PREDICT Q-values instead of storing them.
 * 
 * TWO KEY INNOVATIONS:
 * 
 * 1. EXPERIENCE REPLAY
 * --------------------
 * Instead of learning from each experience once and forgetting it,
 * we store experiences in a "memory" and learn from random batches.
 * 
 * Why? 
 * - Prevents the network from forgetting past lessons
 * - Breaks correlations between consecutive experiences
 * - Like studying for a test by reviewing old homework
 * 
 * 2. TARGET NETWORK
 * -----------------
 * We use TWO neural networks:
 * - Main network: Learns and updates every step
 * - Target network: Used to calculate targets, updates slowly
 * 
 * Why?
 * If you use the same network to both predict and calculate targets,
 * you create a feedback loop. The network chases its own predictions.
 * Using a separate target network stabilizes training.
 * 
 * ============================================================================
 * EXPLORATION VS EXPLOITATION (Epsilon-Greedy)
 * ============================================================================
 * 
 * EXPLORATION: Taking random actions to discover new strategies
 * EXPLOITATION: Using what you already know works best
 * 
 * Epsilon (ε) controls this balance:
 * - ε = 0.3 means: 30% random actions, 70% best actions
 * - High ε = More exploration (trying new things)
 * - Low ε = More exploitation (using what works)
 * 
 * We start with HIGH epsilon (0.3) and DECAY it over time.
 * This is like a child learning: tries many things at first,
 * then focuses on what works as they gain experience.
 * 
 * ============================================================================
 * ADAPTIVE FEATURES IN THIS CODE
 * ============================================================================
 * 
 * 1. ADAPTIVE ENVIRONMENT
 *    The hill moves at different speeds over time (40-160 steps between moves)
 *    The agent must continuously adapt!
 * 
 * 2. ADAPTIVE EXPLORATION
 *    Epsilon decays automatically: ε = ε × 0.995 each episode
 *    Explores less as it learns more
 * 
 * 3. ADAPTIVE REWARDS
 *    Early after hill moves: Reward SPEED (reach new hill fast)
 *    Later: Reward STAYING (remain in hill)
 * 
 * ============================================================================
 * HOW TO USE THIS MODULE
 * ============================================================================
 * 
 * STEP 1: Include TensorFlow.js in your HTML:
 * <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.15.0/dist/tf.min.js"></script>
 * 
 * STEP 2: Include this file:
 * <script src="tensorflow-adaptive-rl.js"></script>
 * 
 * STEP 3: Create the agent:
 * const agent = new TensorFlowDQNAgent();
 * 
 * STEP 4: In your game loop:
 * // Get action from agent
 * const action = await agent.getAction(agentPos, hillPos, stepsSinceLastMove, interval, inZone);
 * 
 * // Take action in game
 * // Get reward
 * 
 * // Train the agent
 * const obs = agent.getObservation(agentPos, hillPos, steps, interval, inZone);
 * const nextObs = agent.getObservation(newAgentPos, hillPos, newSteps, interval, newInZone);
 * await agent.train(obs, action, reward, nextObs, false);
 * 
 * ============================================================================
 * UNDERSTANDING THE OUTPUT (What the Numbers Mean)
 * ============================================================================
 * 
 * Watching the AI learn (console output):
 * 
 * Episode 1/20 | Reward: 45.2
 * - AI is still random (ε = 0.30)
 * - Gets about 45 points in a 300-step episode
 * 
 * Episode 10/20 | Reward: 68.3
 * - AI is learning (ε = 0.22)
 * - Getting better at finding the hill
 * 
 * Episode 20/20 | Reward: 85.6
 * - AI is smart (ε = 0.08)
 * - Good at chasing the moving hill
 * 
 * What GOOD looks like:
 * - Episode Reward INCREASING over time
 * - Epsilon (ε) DECREASING over time
 * - Agent finds the hill FASTER after it moves
 * 
 * ============================================================================
 * TROUBLESHOOTING
 * ============================================================================
 * 
 * Q: AI never learns / stays random
 * A: Make sure you're training enough episodes (20-50 minimum)
 * 
 * Q: Game freezes during training
 * A: Add small delays between episodes (we have 20ms delay)
 * 
 * Q: Agent always takes same action
 * A: Check if epsilon is too low (below 0.05)
 * 
 * Q: Network loss is NaN (Not a Number)
 * A: Check that rewards are within reasonable range (-1 to 1)
 * 
 * ============================================================================
 * CLASS DOCUMENTATION
 * ============================================================================
 * 
 * @class TensorFlowDQNAgent
 * @description Deep Q-Network agent for adaptive RL
 * 
 * @method constructor(config)
 * @param {Object} config - Optional configuration
 * @param {number} config.epsilon - Initial exploration rate (default: 0.3)
 * @param {number} config.memorySize - Experience replay buffer size (default: 3000)
 * 
 * @method getObservation(pos, hill, steps, interval, inZone)
 * @param {Object} pos - Agent position {x, z}
 * @param {Object} hill - Hill position {x, z}
 * @param {number} steps - Steps since last hill move
 * @param {number} interval - Current hill move interval
 * @param {boolean} inZone - Whether agent is in hill zone
 * @returns {number[]} Array of 6 normalized observation values
 * 
 * @method getAction(pos, hill, steps, interval, inZone)
 * @async
 * @returns {Promise<number>} Action 0-4 (Up, Down, Left, Right, Stay)
 * 
 * @method train(obs, action, reward, nextObs, done)
 * @async
 * @param {number[]} obs - Current observation
 * @param {number} action - Action taken
 * @param {number} reward - Reward received
 * @param {number[]} nextObs - Next observation
 * @param {boolean} done - Whether episode ended
 * 
 * @method incrementEpisode()
 * @description Increments episode counter and updates UI
 * 
 * @method updateUI()
 * @description Updates HTML elements with current metrics
 * 
 * @method getEpsilon()
 * @returns {number} Current epsilon value
 * 
 * @method reset()
 * @description Resets the agent for a new training session
 * 
 * ============================================================================
 */

class TensorFlowDQNAgent {
    /**
     * Creates a new DQN Agent
     * 
     * The constructor builds the neural network and initializes
     * the experience replay memory.
     * 
     * NEURAL NETWORK ARCHITECTURE:
     * - Input layer: 6 neurons (observation values)
     * - Hidden layer 1: 64 neurons with ReLU activation
     * - Hidden layer 2: 64 neurons with ReLU activation
     * - Output layer: 5 neurons (Q-values for each action)
     * 
     * @param {Object} config - Configuration options
     * @param {number} config.epsilon - Initial exploration rate (default: 0.3)
     * @param {number} config.memorySize - Size of experience replay buffer (default: 3000)
     */
    constructor(config = {}) {
        // EXPLORATION RATE (epsilon)
        // Controls how often the agent takes random actions vs learned actions
        // 0.3 = 30% random, 70% learned
        this.epsilon = config.epsilon || 0.3;
        
        // EPISODE COUNTER
        // Tracks how many training episodes have been completed
        this.episodes = 0;
        
        // EXPERIENCE REPLAY MEMORY
        // Stores past experiences (state, action, reward, next state)
        // The agent learns from random batches of these experiences
        this.memory = [];
        this.memorySize = config.memorySize || 3000;
        
        // ============================================
        // BUILD THE NEURAL NETWORK
        // ============================================
        // 
        // tf.sequential() creates a stack of layers where data flows
        // from input to output through each layer in sequence.
        // 
        // LAYER 1: Dense(64, ReLU, inputShape=[6])
        // - "Dense" means every neuron connects to every neuron in next layer
        // - 64 neurons in this layer
        // - ReLU activation: f(x) = max(0, x)
        // - inputShape=[6] means this layer receives 6 numbers
        //
        // LAYER 2: Dense(64, ReLU)
        // - Second hidden layer, also with 64 neurons
        // - Takes input from first hidden layer
        // - Learns more complex patterns
        //
        // LAYER 3: Dense(5, linear)
        // - Output layer with 5 neurons (one for each action)
        // - Linear activation means output is not transformed
        // - Values are the Q-estimates for each action
        //
        this.model = tf.sequential();
        this.model.add(tf.layers.dense({ 
            units: 64, 
            activation: 'relu', 
            inputShape: [6] 
        }));
        this.model.add(tf.layers.dense({ 
            units: 64, 
            activation: 'relu' 
        }));
        this.model.add(tf.layers.dense({ 
            units: 5, 
            activation: 'linear' 
        }));
        
        // COMPILE THE MODEL
        // - optimizer: 'adam' - Adapts learning rate automatically
        // - loss: 'meanSquaredError' - Measures how wrong predictions are
        this.model.compile({ 
            optimizer: 'adam', 
            loss: 'meanSquaredError' 
        });
        
        console.log('========================================');
        console.log('TensorFlow DQN Agent Initialized');
        console.log('Network Architecture: 6 → 64 → 64 → 5');
        console.log('Learning Rate: Adam (adaptive)');
        console.log('Exploration (ε):', this.epsilon);
        console.log('Memory Size:', this.memorySize);
        console.log('========================================');
    }
    
    /**
     * Get observation vector from game state
     * 
     * The observation is what the agent can "see" about the game.
     * It's a set of 6 numbers that describe the current situation:
     * 
     * Index 0: Agent X position (normalized -1 to 1)
     * Index 1: Agent Z position (normalized -1 to 1)
     * Index 2: Hill X position (normalized -1 to 1)
     * Index 3: Hill Z position (normalized -1 to 1)
     * Index 4: Time until hill moves (0 = just moved, 1 = about to move)
     * Index 5: Is agent on the hill? (0 = no, 1 = yes)
     * 
     * NORMALIZATION explained:
     * Positions range from -18 to +18 (arena size 36)
     * Dividing by 18 gives values from -1 to +1
     * This helps the neural network learn because all inputs are on the same scale
     * 
     * @param {Object} pos - Agent position {x, z}
     * @param {Object} hill - Hill position {x, z}
     * @param {number} steps - Steps since last hill move
     * @param {number} interval - Current hill move interval
     * @param {boolean} inZone - Whether agent is in the hill zone
     * @returns {number[]} Observation array of 6 numbers
     */
    getObservation(pos, hill, steps, interval, inZone) {
        // Normalize positions by dividing by arena half-size (18)
        const normAgentX = pos.x / 18;
        const normAgentZ = pos.z / 18;
        const normHillX = hill.x / 18;
        const normHillZ = hill.z / 18;
        
        // Time normalization: steps / interval, clamped to max 1
        // 0 = hill just moved, 1 = hill about to move
        const timeNorm = Math.min(1, steps / interval);
        
        // Binary flag: 1 if in zone, 0 if not
        const inZoneFlag = inZone ? 1 : 0;
        
        return [normAgentX, normAgentZ, normHillX, normHillZ, timeNorm, inZoneFlag];
    }
    
    /**
     * Get the best action for current state using epsilon-greedy policy
     * 
     * EPSILON-GREEDY EXPLANATION:
     * ---------------------------
     * With probability ε (epsilon): Take a RANDOM action (EXPLORE)
     * With probability 1-ε: Take the BEST action according to network (EXPLOIT)
     * 
     * EXPLORATION (Random actions):
     * - Tries new things the agent hasn't done before
     * - Might find better strategies
     * - Important early in training
     * 
     * EXPLOITATION (Best actions):
     * - Uses what the agent has already learned
     * - Gets consistent rewards
     * - Important after training
     * 
     * The epsilon value starts high (0.3) and decays over time.
     * This is like a child learning: tries many things at first,
     * then focuses on what works as they gain experience.
     * 
     * @param {Object} pos - Agent position
     * @param {Object} hill - Hill position
     * @param {number} steps - Steps since last hill move
     * @param {number} interval - Current hill move interval
     * @param {boolean} inZone - Whether agent is in zone
     * @returns {Promise<number>} Action (0-4)
     * 
     * ACTIONS MAPPING:
     * 0 = Move UP (increase Z)
     * 1 = Move DOWN (decrease Z)
     * 2 = Move LEFT (decrease X)
     * 3 = Move RIGHT (increase X)
     * 4 = STAY (don't move)
     */
    async getAction(pos, hill, steps, interval, inZone) {
        // STEP 1: Get the current observation (what the agent sees)
        const obs = this.getObservation(pos, hill, steps, interval, inZone);
        
        // STEP 2: Convert observation to TensorFlow tensor
        // TensorFlow uses tensors (multi-dimensional arrays) for calculations
        // [1, 6] means: 1 row, 6 columns (batch size 1, 6 features)
        const input = tf.tensor2d([obs], [1, 6]);
        
        // STEP 3: Forward pass through neural network
        // This gives Q-values for each action
        const qValues = this.model.predict(input);
        
        // STEP 4: Get the actual numbers from the tensor
        const values = await qValues.data();
        
        // STEP 5: Clean up tensors (prevent memory leaks!)
        input.dispose();
        qValues.dispose();
        
        // STEP 6: Epsilon-greedy action selection
        if (Math.random() < this.epsilon) {
            // EXPLORE: Take random action
            return Math.floor(Math.random() * 5);
        } else {
            // EXPLOIT: Take action with highest Q-value
            // values.indexOf(Math.max(...values)) finds the index of the largest number
            return values.indexOf(Math.max(...values));
        }
    }
    
    /**
     * Train the neural network on a single experience
     * 
     * This function implements the DQN learning algorithm:
     * 1. Store the experience in memory (experience replay)
     * 2. Sample random batch from memory
     * 3. Calculate target Q-values using Bellman equation
     * 4. Train the network to predict these targets
     * 5. Decay epsilon (exploration rate)
     * 
     * BELLMAN EQUATION IN CODE:
     * if (done) {
     *     target = reward
     * } else {
     *     target = reward + gamma * max(next Q-values)
     * }
     * 
     * @param {number[]} obs - Current observation
     * @param {number} action - Action taken
     * @param {number} reward - Reward received
     * @param {number[]} nextObs - Next observation
     * @param {boolean} done - Whether episode ended
     */
    async train(obs, action, reward, nextObs, done) {
        // ============================================
        // STEP 1: Store experience in memory
        // ============================================
        // Experience replay stores past experiences so we can learn from them
        // multiple times. This prevents catastrophic forgetting.
        this.memory.push({ obs, action, reward, nextObs, done });
        
        // Limit memory size (remove oldest if too big)
        if (this.memory.length > this.memorySize) {
            this.memory.shift();  // Remove first (oldest) element
        }
        
        // Need at least 32 experiences to train (batch size)
        if (this.memory.length < 32) return;
        
        // ============================================
        // STEP 2: Sample random batch from memory
        // ============================================
        // Random sampling breaks correlations between consecutive experiences.
        // This makes learning more stable.
        const batch = [];
        const indices = new Set();
        
        // Collect 32 random indices from memory
        while (indices.size < 32) {
            indices.add(Math.floor(Math.random() * this.memory.length));
        }
        
        // Build the batch from these indices
        for (const idx of indices) {
            batch.push(this.memory[idx]);
        }
        
        // ============================================
        // STEP 3: Prepare batch data for training
        // ============================================
        const states = batch.map(e => e.obs);        // Current states
        const nextStates = batch.map(e => e.nextObs); // Next states
        
        // ============================================
        // STEP 4: Get current Q-values from MAIN network
        // ============================================
        const currentQTensor = this.model.predict(tf.tensor2d(states, [32, 6]));
        const currentQ = await currentQTensor.array();
        
        // ============================================
        // STEP 5: Get next Q-values from MAIN network
        // ============================================
        const nextQTensor = this.model.predict(tf.tensor2d(nextStates, [32, 6]));
        const nextQ = await nextQTensor.array();
        
        // ============================================
        // STEP 6: Calculate target Q-values using Bellman equation
        // ============================================
        // For each experience in the batch:
        // - If episode ended: target = reward
        // - If not ended: target = reward + gamma * max(next Q)
        const targets = [];
        
        for (let i = 0; i < 32; i++) {
            // Start with current Q-values (copy array)
            const target = [...currentQ[i]];
            const a = batch[i].action;    // Action taken
            const r = batch[i].reward;    // Reward received
            const d = batch[i].done;      // Whether episode ended
            
            if (d) {
                // Terminal state: target is just the reward
                target[a] = r;
            } else {
                // Non-terminal: target = reward + gamma * max(next Q)
                // gamma = 0.95 (discount factor)
                // max(next Q) = best possible future reward
                const maxNextQ = Math.max(...nextQ[i]);
                target[a] = r + 0.95 * maxNextQ;
            }
            targets.push(target);
        }
        
        // ============================================
        // STEP 7: Train the network
        // ============================================
        // We give the network: input = states, target = targets
        // The network learns to predict the targets
        const xs = tf.tensor2d(states, [32, 6]);
        const ys = tf.tensor2d(targets, [32, 5]);
        await this.model.fit(xs, ys, { epochs: 1, verbose: 0 });
        
        // ============================================
        // STEP 8: Clean up tensors (prevent memory leaks!)
        // ============================================
        xs.dispose();
        ys.dispose();
        currentQTensor.dispose();
        nextQTensor.dispose();
        
        // ============================================
        // STEP 9: Decay epsilon (exploration rate)
        // ============================================
        // Multiply by 0.995 each episode
        // Example: 0.3 → 0.2985 → 0.297 → ... → 0.05
        // This reduces random actions over time as the agent learns
        this.epsilon = Math.max(0.05, this.epsilon * 0.995);
    }
    
    /**
     * Increment episode counter and update UI
     * Call this after each training episode completes
     */
    incrementEpisode() {
        this.episodes++;
        this.updateUI();
    }
    
    /**
     * Update UI elements with current metrics
     * Shows the user:
     * - Current epsilon (exploration rate)
     * - Number of training episodes completed
     * - Memory size (how many experiences stored)
     */
    updateUI() {
        const epsilonEl = document.getElementById('epsilon');
        const episodesEl = document.getElementById('train-episodes');
        const memoryEl = document.getElementById('memory');
        
        if (epsilonEl) epsilonEl.textContent = this.epsilon.toFixed(3);
        if (episodesEl) episodesEl.textContent = this.episodes;
        if (memoryEl) memoryEl.textContent = this.memory.length;
    }
    
    /**
     * Get current epsilon value
     * @returns {number} Current exploration rate (0-1)
     */
    getEpsilon() {
        return this.epsilon;
    }
    
    /**
     * Reset the agent for a new training session
     * Resets epsilon, episode counter, and memory
     */
    reset() {
        this.epsilon = 0.3;
        this.episodes = 0;
        this.memory = [];
        this.updateUI();
        console.log('Agent reset. Epsilon:', this.epsilon);
    }
}

// Make the class available globally so HTML can use it
window.TensorFlowDQNAgent = TensorFlowDQNAgent;

console.log('========================================');
console.log('TensorFlow.js Adaptive RL Module Loaded');
console.log('========================================');
console.log('Class available: TensorFlowDQNAgent');
console.log('');
console.log('QUICK START:');
console.log('  const agent = new TensorFlowDQNAgent();');
console.log('  const action = await agent.getAction(agentPos, hillPos, steps, interval, inZone);');
console.log('  await agent.train(obs, action, reward, nextObs, done);');
console.log('========================================');
