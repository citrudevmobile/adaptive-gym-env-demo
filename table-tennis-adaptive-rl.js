/**
 * ============================================================================
 * TABLE TENNIS - ADAPTIVE REINFORCEMENT LEARNING MODULE
 * ============================================================================
 * 
 * A complete Deep Q-Network (DQN) implementation for Table Tennis.
 * This module contains the AI agent that learns to play using a neural network.
 * 
 * ============================================================================
 * WHY TABLE TENNIS IS HARDER THAN KING OF THE HILL
 * ============================================================================
 * 
 * King of the Hill (Version 2):
 * - Agent moves in 2D plane
 * - Goal is static between moves
 * - Simple distance-based reward
 * - Agent can just "go to" the hill
 * 
 * Table Tennis (Version 3):
 * - Agent must track a moving ball (3D trajectory!)
 * - Must predict where the ball will land
 * - Timing is critical (hit too early/late = miss)
 * - Ball physics with gravity and bounce
 * - Opponent returns ball with different trajectories
 * 
 * ============================================================================
 * STATE SPACE (What the agent sees)
 * ============================================================================
 * 
 * The agent receives 14 numbers describing the game state:
 * 
 * Index 0: Ball X position (normalized -1 to 1)
 * Index 1: Ball Y position (normalized -1 to 1)
 * Index 2: Ball Z position (normalized -1 to 1)
 * Index 3: Ball X velocity (how fast sideways)
 * Index 4: Ball Y velocity (how fast up/down)
 * Index 5: Ball Z velocity (how fast toward/away)
 * Index 6: AI Paddle X position (left/right)
 * Index 7: AI Paddle Y position (up/down)
 * Index 8: Opponent Paddle X position
 * Index 9: Predicted landing X (where ball will hit)
 * Index 10: Predicted landing Z
 * Index 11: Time until ball reaches paddle (normalized)
 * Index 12: Last hit type (encoded 0-6)
 * Index 13: Rally length (how many hits so far)
 * 
 * ============================================================================
 * ACTION SPACE (What the agent can do)
 * ============================================================================
 * 
 * The agent can choose from 9 actions:
 * 
 * 0 = Move paddle LEFT
 * 1 = Move paddle RIGHT
 * 2 = Move paddle UP
 * 3 = Move paddle DOWN
 * 4 = Hit FLAT (standard return)
 * 5 = Hit TOPSPIN (ball dips faster after bounce)
 * 6 = Hit BACKSPIN (ball floats, stays low)
 * 7 = Hit SMASH (powerful fast shot)
 * 8 = Hit LOB (high, deep defensive shot)
 * 
 * ============================================================================
 * REWARD STRUCTURE (What tells the agent it did well)
 * ============================================================================
 * 
 * - Win rally: +1.0
 * - Lose rally: -1.0
 * - Hit ball over net: +0.1
 * - Force opponent error: +0.2
 * - Make unforced error: -0.3
 * - Hit to corner: +0.05
 * - Return with spin: +0.05
 * - Long rally: +0.01 per hit (rewards consistency)
 * 
 * ============================================================================
 * NEURAL NETWORK ARCHITECTURE
 * ============================================================================
 * 
 *     INPUT LAYER      HIDDEN LAYER 1   HIDDEN LAYER 2   HIDDEN LAYER 3    OUTPUT LAYER
 *    (14 neurons)        (128 neurons)     (128 neurons)     (64 neurons)     (9 neurons)
 *         │                    │                │                │              │
 *    ┌────┼────┐          ┌─────┼─────┐     ┌─────┼─────┐     ┌─────┼─────┐    ┌─────┼─────┐
 *    │    │    │          │     │     │     │     │     │     │     │     │    │     │     │
 *    ▼    ▼    ▼          ▼     ▼     ▼     ▼     ▼     ▼     ▼     ▼     ▼    ▼     ▼     ▼
 *   [●]  [●]  [●]   →   [●●]  [●●]  [●●]  →  [●●]  [●●]  [●●]  → [●●]  [●●]  [●●] → [●]  [●]  [●]
 *    │    │    │          │     │     │     │     │     │     │     │     │    │     │     │
 *    └────┼────┘          └─────┼─────┘     └─────┼─────┘     └─────┼─────┘    └─────┼─────┘
 *         │                    │                │                │              │
 *    Ball position         Learns basic      Learns complex    Learns timing    Q-values for:
 *    Ball velocity         patterns like     patterns like     patterns like    0 = Move Left
 *    Paddle positions      "move toward      "if ball is       "hit when        1 = Move Right
 *    Landing prediction    ball"             far, move         ball is close"   2 = Move Up
 *    Time to hit                             faster"                            3 = Move Down
 *    Rally history                                                 4 = Hit Flat
 *                                                                  5 = Hit Topspin
 *                                                                  6 = Hit Backspin
 *                                                                  7 = Hit Smash
 *                                                                  8 = Hit Lob
 * 
 * ============================================================================
 * KEY DQN CONCEPTS
 * ============================================================================
 * 
 * 1. EXPERIENCE REPLAY
 *    Stores past experiences in memory and learns from random batches.
 *    Prevents forgetting and breaks correlations between consecutive plays.
 * 
 * 2. EPSILON-GREEDY EXPLORATION
 *    With probability epsilon: take random action (explore)
 *    With probability 1-epsilon: take best action (exploit)
 *    Epsilon decays over time: starts at 0.3, ends at 0.05
 * 
 * 3. BELLMAN EQUATION
 *    Q(s,a) = Q(s,a) + α × [ R + γ × max(Q(s',a')) - Q(s,a) ]
 * 
 * 4. DISCOUNT FACTOR (gamma = 0.95)
 *    Values future rewards almost as much as immediate rewards.
 *    Encourages long-term strategy.
 * 
 * ============================================================================
 * HOW TO USE THIS MODULE
 * ============================================================================
 * 
 * STEP 1: Include TensorFlow.js in your HTML:
 * <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.15.0/dist/tf.min.js"></script>
 * 
 * STEP 2: Include this file:
 * <script src="table-tennis-rl.js"></script>
 * 
 * STEP 3: Create the agent:
 * const agent = new TableTennisAgent();
 * 
 * STEP 4: In your game loop, get observation and action:
 * const observation = agent.getObservation(ballPos, ballVel, aiPaddle, opponentPaddle, landing, timeToHit, hitType, rallyLength);
 * const action = await agent.getAction(observation);
 * 
 * STEP 5: Apply action to game, get reward, then train:
 * await agent.train(observation, action, reward, nextObservation, done);
 * 
 * ============================================================================
 */

class TableTennisAgent {
    /**
     * Creates a DQN agent for table tennis
     * 
     * NEURAL NETWORK ARCHITECTURE:
     * - Input layer: 14 neurons (ball position, velocity, paddle positions, etc.)
     * - Hidden layer 1: 128 neurons with ReLU activation
     * - Hidden layer 2: 128 neurons with ReLU activation
     * - Hidden layer 3: 64 neurons with ReLU activation
     * - Output layer: 9 neurons (Q-values for each action)
     * 
     * WHY LARGER NETWORK?
     * Table tennis is more complex than King of the Hill:
     * - More state variables (14 vs 6)
     * - More actions (9 vs 5)
     * - Requires timing and prediction
     * - Physics simulation adds complexity
     * 
     * @param {Object} config - Configuration options
     * @param {number} config.epsilon - Initial exploration rate (default: 0.3)
     * @param {number} config.memorySize - Experience replay buffer size (default: 5000)
     * @param {number} config.learningRate - Learning rate (default: 0.001)
     * @param {number} config.discountFactor - Discount factor gamma (default: 0.95)
     */
    constructor(config = {}) {
        // Learning parameters
        this.epsilon = config.epsilon || 0.3;           // Exploration rate (0.3 = 30% random actions)
        this.epsilonMin = config.epsilonMin || 0.05;    // Minimum exploration rate
        this.epsilonDecay = config.epsilonDecay || 0.995; // How fast epsilon decays
        this.discountFactor = config.discountFactor || 0.95; // Gamma - how much to value future rewards
        this.learningRate = config.learningRate || 0.001;    // Alpha - how fast to learn
        
        // Training tracking
        this.episodes = 0;           // Number of training episodes completed
        this.totalSteps = 0;         // Total actions taken
        this.trainingLoss = 0;       // Last training loss value
        
        // Experience replay memory
        this.memory = [];
        this.memorySize = config.memorySize || 5000;
        this.batchSize = config.batchSize || 64;      // Larger batch for complex game
        
        // Target network update frequency
        this.targetUpdateFreq = config.targetUpdateFreq || 100;
        this.trainStep = 0;
        
        // Adaptive strategy tracking
        this.learningStyle = "balanced";   // aggressive, defensive, balanced
        this.opponentPatterns = {           // Tracks opponent habits
            leftHits: 0,
            rightHits: 0,
            smashCount: 0,
            totalHits: 0
        };
        
        // ================================================================
        // BUILD THE NEURAL NETWORK
        // ================================================================
        // 
        // WHY 3 HIDDEN LAYERS?
        // Table tennis requires learning complex patterns:
        // - Layer 1: Basic ball tracking
        // - Layer 2: Timing and prediction
        // - Layer 3: Strategy and adaptation
        //
        // WHY 128 NEURONS?
        // More neurons = more learning capacity
        // 128 is a good balance for this problem size
        //
        
        this.model = this._buildNetwork();
        this.targetModel = this._buildNetwork();  // Target network for stable training
        this._updateTargetNetwork();               // Copy weights to target
        
        // Optimizer for training
        this.optimizer = tf.train.adam(this.learningRate);
        
        console.log('========================================');
        console.log('🏓 Table Tennis DQN Agent Initialized');
        console.log('========================================');
        console.log('Network Architecture: 14 → 128 → 128 → 64 → 9');
        console.log('Trainable Parameters:', this._countParameters());
        console.log('Exploration Rate (ε):', this.epsilon);
        console.log('Memory Size:', this.memorySize);
        console.log('Batch Size:', this.batchSize);
        console.log('Discount Factor (γ):', this.discountFactor);
        console.log('========================================');
    }
    
    /**
     * Build the neural network model
     * @returns {tf.Sequential} TensorFlow.js sequential model
     */
    _buildNetwork() {
        const model = tf.sequential();
        
        // Hidden layer 1: 128 neurons with ReLU activation
        // ReLU = Rectified Linear Unit: f(x) = max(0, x)
        // Turns off negative values, helps learn non-linear patterns
        model.add(tf.layers.dense({
            units: 128,
            activation: 'relu',
            inputShape: [14]
        }));
        
        // Hidden layer 2: 128 neurons with ReLU activation
        model.add(tf.layers.dense({
            units: 128,
            activation: 'relu'
        }));
        
        // Hidden layer 3: 64 neurons with ReLU activation
        model.add(tf.layers.dense({
            units: 64,
            activation: 'relu'
        }));
        
        // Output layer: 9 neurons (one per action) with linear activation
        // Linear activation means output is not transformed (raw Q-values)
        model.add(tf.layers.dense({
            units: 9,
            activation: 'linear'
        }));
        
        return model;
    }
    
    /**
     * Count total trainable parameters in the network
     * @returns {number} Number of parameters
     */
    _countParameters() {
        let count = 0;
        const weights = this.model.getWeights();
        for (const w of weights) {
            const shape = w.shape;
            let params = 1;
            for (const s of shape) params *= s;
            count += params;
        }
        return count;
    }
    
    /**
     * Update target network weights from main network
     * Target network is a "slower" copy used for stable training
     */
    _updateTargetNetwork() {
        const weights = this.model.getWeights();
        this.targetModel.setWeights(weights);
    }
    
    /**
     * Get observation vector from game state
     * 
     * NORMALIZATION explained:
     * Positions are divided by max values to get range -1 to 1
     * This helps the neural network learn because all inputs are on the same scale
     * 
     * @param {Object} ballPos - Ball position {x, y, z}
     * @param {Object} ballVel - Ball velocity {x, y, z}
     * @param {Object} aiPaddle - AI paddle position {x, y}
     * @param {Object} opponentPaddle - Opponent paddle position {x, y}
     * @param {Object} predictedLanding - Where ball will land {x, z}
     * @param {number} timeToHit - Frames until ball reaches paddle
     * @param {number} lastHitType - Type of last hit (0-8)
     * @param {number} rallyLength - Number of hits in current rally
     * @returns {number[]} Array of 14 normalized observation values
     */
    getObservation(ballPos, ballVel, aiPaddle, opponentPaddle, predictedLanding, timeToHit, lastHitType, rallyLength) {
        // Normalization constants
        const MAX_POS = 5;      // Max position (table half-length)
        const MAX_VEL = 15;     // Max velocity (fastest ball speed)
        const MAX_TIME = 30;    // Max time to hit (frames)
        const MAX_RALLY = 50;   // Max rally length for normalization
        
        return [
            // Ball position (normalized -1 to 1)
            ballPos.x / MAX_POS,
            ballPos.y / 3,           // Ball height max ~3
            ballPos.z / MAX_POS,
            
            // Ball velocity (normalized -1 to 1)
            ballVel.x / MAX_VEL,
            ballVel.y / MAX_VEL,
            ballVel.z / MAX_VEL,
            
            // AI paddle position (normalized -1 to 1)
            aiPaddle.x / 2.5,
            aiPaddle.y / 1.5,
            
            // Opponent paddle position (normalized -1 to 1)
            opponentPaddle.x / 2.5,
            
            // Predicted landing position (normalized -1 to 1)
            predictedLanding.x / MAX_POS,
            predictedLanding.z / MAX_POS,
            
            // Time to hit (normalized 0 to 1)
            Math.min(1, timeToHit / MAX_TIME),
            
            // Last hit type (normalized 0 to 1)
            lastHitType / 8,
            
            // Rally length (normalized 0 to 1)
            Math.min(1, rallyLength / MAX_RALLY)
        ];
    }
    
    /**
     * Get Q-values for an observation (forward pass through network)
     * @param {number[]} observation - Array of 14 numbers
     * @returns {Promise<number[]>} Array of 9 Q-values
     */
    async _getQValues(observation) {
        const inputTensor = tf.tensor2d([observation], [1, 14]);
        const qValues = this.model.predict(inputTensor);
        const values = await qValues.data();
        
        // Clean up tensors to prevent memory leaks
        inputTensor.dispose();
        qValues.dispose();
        
        return Array.from(values);
    }
    
    /**
     * Select action using epsilon-greedy policy
     * 
     * EPSILON-GREEDY EXPLANATION:
     * - With probability ε: take RANDOM action (EXPLORE)
     * - With probability 1-ε: take BEST action according to network (EXPLOIT)
     * 
     * ADAPTIVE FEATURE: When performing poorly, epsilon is higher
     * 
     * @param {number[]} observation - Current observation
     * @returns {Promise<number>} Action (0-8)
     */
    async getAction(observation) {
        // Explore: take random action
        if (Math.random() < this.epsilon) {
            return Math.floor(Math.random() * 9);
        }
        
        // Exploit: take best action according to neural network
        const qValues = await this._getQValues(observation);
        return qValues.indexOf(Math.max(...qValues));
    }
    
    /**
     * Calculate reward for an action
     * 
     * REWARD DESIGN EXPLANATION:
     * - Win rally: Large positive reward (+1.0)
     * - Lose rally: Large negative reward (-1.0)
     * - Hit ball: Small positive reward (+0.1) encourages hitting
     * - Forced error: Extra reward (+0.2) for good shots
     * - Unforced error: Extra penalty (-0.3) for bad shots
     * - Corner hit: Small bonus (+0.05) for placement
     * - Spin shot: Small bonus (+0.05) for variety
     * - Long rally: Small bonus for consistency
     * 
     * @param {Object} result - Result of the action
     * @returns {number} Reward value
     */
    calculateReward(result) {
        let reward = 0;
        
        if (result.winRally) {
            reward += 1.0;
        } else if (result.loseRally) {
            reward -= 1.0;
        }
        
        if (result.hitMade) {
            reward += 0.1;
        }
        
        if (result.forcedError) {
            reward += 0.2;
        }
        
        if (result.unforcedError) {
            reward -= 0.3;
        }
        
        if (result.hitToCorner) {
            reward += 0.05;
        }
        
        if (result.usedSpin) {
            reward += 0.05;
        }
        
        // Small reward for long rallies (consistency)
        if (result.rallyLength > 5) {
            reward += 0.01;
        }
        
        return reward;
    }
    
    /**
     * Train the neural network on a single experience
     * 
     * This implements the DQN learning algorithm:
     * 1. Store experience in memory (experience replay)
     * 2. Sample random batch from memory
     * 3. Calculate target Q-values using Bellman equation
     * 4. Train the network to predict these targets
     * 5. Update target network periodically
     * 6. Decay epsilon (exploration rate)
     * 
     * @param {number[]} observation - Current observation
     * @param {number} action - Action taken (0-8)
     * @param {number} reward - Reward received
     * @param {number[]} nextObservation - Next observation
     * @param {boolean} done - Whether episode ended
     * @returns {Promise<void>}
     */
    async train(observation, action, reward, nextObservation, done) {
        // ================================================================
        // STEP 1: Store experience in memory
        // ================================================================
        this.memory.push({
            observation: observation.slice(),  // Copy to avoid reference issues
            action: action,
            reward: reward,
            nextObservation: nextObservation.slice(),
            done: done
        });
        
        // Limit memory size (remove oldest if too big)
        if (this.memory.length > this.memorySize) {
            this.memory.shift();
        }
        
        // Need minimum batch size to train
        if (this.memory.length < this.batchSize) return;
        
        // ================================================================
        // STEP 2: Sample random batch from memory
        // ================================================================
        const batch = [];
        const indices = new Set();
        while (indices.size < this.batchSize) {
            indices.add(Math.floor(Math.random() * this.memory.length));
        }
        for (const idx of indices) batch.push(this.memory[idx]);
        
        // ================================================================
        // STEP 3: Prepare batch data
        // ================================================================
        const states = batch.map(e => e.observation);
        const nextStates = batch.map(e => e.nextObservation);
        
        // ================================================================
        // STEP 4: Get current Q-values from MAIN network
        // ================================================================
        const statesTensor = tf.tensor2d(states, [this.batchSize, 14]);
        const currentQTensor = this.model.predict(statesTensor);
        const currentQ = await currentQTensor.array();
        
        // ================================================================
        // STEP 5: Get next Q-values from TARGET network
        // (Using target network stabilizes training)
        // ================================================================
        const nextStatesTensor = tf.tensor2d(nextStates, [this.batchSize, 14]);
        const nextQTensor = this.targetModel.predict(nextStatesTensor);
        const nextQ = await nextQTensor.array();
        
        // ================================================================
        // STEP 6: Calculate target Q-values using Bellman equation
        // ================================================================
        const targets = [];
        for (let i = 0; i < this.batchSize; i++) {
            const target = [...currentQ[i]];
            const a = batch[i].action;
            const r = batch[i].reward;
            const d = batch[i].done;
            
            if (d) {
                // Terminal state: target is just the reward
                target[a] = r;
            } else {
                // Bellman equation: target = reward + gamma * max(next Q)
                const maxNextQ = Math.max(...nextQ[i]);
                target[a] = r + this.discountFactor * maxNextQ;
            }
            targets.push(target);
        }
        
        // ================================================================
        // STEP 7: Train the network (gradient descent)
        // ================================================================
        const targetsTensor = tf.tensor2d(targets, [this.batchSize, 9]);
        
        const loss = await this._trainBatch(statesTensor, targetsTensor);
        this.trainingLoss = loss;
        
        // ================================================================
        // STEP 8: Clean up tensors (prevent memory leaks!)
        // ================================================================
        statesTensor.dispose();
        nextStatesTensor.dispose();
        currentQTensor.dispose();
        nextQTensor.dispose();
        targetsTensor.dispose();
        
        // ================================================================
        // STEP 9: Update target network periodically
        // ================================================================
        this.trainStep++;
        if (this.trainStep % this.targetUpdateFreq === 0) {
            this._updateTargetNetwork();
        }
        
        // ================================================================
        // STEP 10: Decay epsilon (exploration rate)
        // ================================================================
        this.epsilon = Math.max(this.epsilonMin, this.epsilon * this.epsilonDecay);
        
        this.totalSteps++;
    }
    
    /**
     * Train the model on a batch
     * @param {tf.Tensor} states - Batch of states
     * @param {tf.Tensor} targets - Target Q-values
     * @returns {Promise<number>} Loss value
     */
    async _trainBatch(states, targets) {
        return new Promise((resolve) => {
            this.optimizer.minimize(() => {
                const predictions = this.model.apply(states, { training: true });
                const loss = tf.losses.meanSquaredError(targets, predictions);
                resolve(loss.dataSync()[0]);
                return loss;
            });
        });
    }
    
    /**
     * End an episode and update tracking
     * @param {number} totalReward - Total reward for the episode
     */
    endEpisode(totalReward) {
        this.episodes++;
        this.updateUI();
        
        // Adaptive strategy adjustment based on performance
        if (totalReward > 50) {
            this.learningStyle = "confident";
        } else if (totalReward < 10) {
            this.learningStyle = "cautious";
            // Boost exploration when doing poorly
            this.epsilon = Math.min(0.4, this.epsilon + 0.05);
        } else {
            this.learningStyle = "balanced";
        }
        
        console.log(`📊 Episode ${this.episodes} complete. Total reward: ${totalReward.toFixed(1)}. ε: ${this.epsilon.toFixed(3)}`);
    }
    
    /**
     * Update opponent pattern tracking (adaptive feature)
     * The AI learns where the opponent likes to hit
     * @param {Object} hit - Information about the opponent's hit
     */
    updateOpponentPattern(hit) {
        this.opponentPatterns.totalHits++;
        
        if (hit.targetX < -1) {
            this.opponentPatterns.leftHits++;
        } else if (hit.targetX > 1) {
            this.opponentPatterns.rightHits++;
        }
        
        if (hit.isSmash) {
            this.opponentPatterns.smashCount++;
        }
        
        // Adapt strategy based on patterns
        const leftRatio = this.opponentPatterns.leftHits / this.opponentPatterns.totalHits;
        const rightRatio = this.opponentPatterns.rightHits / this.opponentPatterns.totalHits;
        
        if (leftRatio > 0.6) {
            this.learningStyle = "defendLeft";
        } else if (rightRatio > 0.6) {
            this.learningStyle = "defendRight";
        } else if (this.opponentPatterns.smashCount > 10) {
            this.learningStyle = "stayBack";
        } else {
            this.learningStyle = "balanced";
        }
    }
    
    /**
     * Predict where the ball will land (for the agent's decision making)
     * This is a simplified physics prediction
     * 
     * @param {Object} ballPos - Current ball position
     * @param {Object} ballVel - Current ball velocity
     * @returns {Object} Predicted landing position {x, z}
     */
    predictLanding(ballPos, ballVel) {
        let px = ballPos.x;
        let py = ballPos.y;
        let pz = ballPos.z;
        let vx = ballVel.x;
        let vy = ballVel.y;
        let vz = ballVel.z;
        const GRAVITY = -14;
        const DT = 0.016;
        
        let steps = 0;
        while (steps < 100 && pz < 4.5 && pz > -4.5) {
            vy += GRAVITY * DT;
            px += vx * DT;
            py += vy * DT;
            pz += vz * DT;
            
            if (py < 0.12) {
                vy = -vy * 0.85;
                py = 0.12;
            }
            steps++;
        }
        
        return { x: px, z: pz };
    }
    
    /**
     * Get the current strategy/style for UI display
     * @returns {string} Current learning style
     */
    getStrategy() {
        const styles = {
            "balanced": "⚖️ Balanced",
            "aggressive": "⚡ Aggressive",
            "defensive": "🛡️ Defensive",
            "defendLeft": "⬅️ Guard Left",
            "defendRight": "➡️ Guard Right",
            "stayBack": "📏 Stay Back",
            "confident": "🎯 Confident",
            "cautious": "🐢 Cautious"
        };
        return styles[this.learningStyle] || "⚖️ Balanced";
    }
    
    /**
     * Update UI elements with current metrics
     */
    updateUI() {
        const epsilonEl = document.getElementById('epsilon');
        const episodesEl = document.getElementById('train-episodes');
        const memoryEl = document.getElementById('memory');
        const lossEl = document.getElementById('loss');
        const styleEl = document.getElementById('ai-style');
        
        if (epsilonEl) epsilonEl.textContent = this.epsilon.toFixed(3);
        if (episodesEl) episodesEl.textContent = this.episodes;
        if (memoryEl) memoryEl.textContent = this.memory.length;
        if (lossEl) lossEl.textContent = this.trainingLoss.toFixed(4);
        if (styleEl) styleEl.textContent = this.getStrategy();
    }
    
    /**
     * Get current epsilon value
     * @returns {number} Current exploration rate
     */
    getEpsilon() {
        return this.epsilon;
    }
    
    /**
     * Reset the agent for a new training session
     */
    reset() {
        this.epsilon = 0.3;
        this.episodes = 0;
        this.totalSteps = 0;
        this.memory = [];
        this.opponentPatterns = {
            leftHits: 0,
            rightHits: 0,
            smashCount: 0,
            totalHits: 0
        };
        this.updateUI();
        console.log('Agent reset. Epsilon:', this.epsilon);
    }
    
    /**
     * Save the model (for later use)
     * @returns {Promise<void>}
     */
    async saveModel() {
        await this.model.save('downloads://table-tennis-model');
        console.log('💾 Model saved!');
    }
    
    /**
     * Load a saved model
     * @param {tf.LayersModel} model - Loaded model
     */
    loadModel(model) {
        this.model = model;
        this.targetModel = this._buildNetwork();
        this._updateTargetNetwork();
        console.log('📂 Model loaded!');
    }
}

// Make the class available globally
window.TableTennisAgent = TableTennisAgent;

console.log('========================================');
console.log('🏓 Table Tennis RL Module Loaded');
console.log('========================================');
console.log('');
console.log('QUICK START:');
console.log('  const agent = new TableTennisAgent();');
console.log('  const obs = agent.getObservation(ballPos, ballVel, aiPaddle, opponentPaddle, landing, timeToHit, hitType, rallyLength);');
console.log('  const action = await agent.getAction(obs);');
console.log('  await agent.train(obs, action, reward, nextObs, done);');
console.log('');
console.log('Available methods:');
console.log('  - getObservation()  : Convert game state to 14 numbers');
console.log('  - getAction()       : Choose best action (epsilon-greedy)');
console.log('  - train()           : Learn from experience');
console.log('  - predictLanding()  : Predict where ball will land');
console.log('  - calculateReward() : Calculate reward from game events');
console.log('  - updateOpponentPattern() : Learn opponent habits');
console.log('========================================');