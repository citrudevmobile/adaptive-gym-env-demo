/**
 * ============================================================================
 * TABLE TENNIS - ADAPTIVE REINFORCEMENT LEARNING MODULE
 * ============================================================================
 * 
 * A complete Deep Q-Network (DQN) implementation for Table Tennis.
 * This module contains the AI agent that learns to play using a neural network.
 * 
 * ============================================================================
 * STATE SPACE (What the agent sees) - 14 dimensions
 * ============================================================================
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
 * ACTION SPACE (What the agent can do) - 9 actions
 * ============================================================================
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
 * NEURAL NETWORK ARCHITECTURE
 * ============================================================================
 * 
 *     INPUT LAYER      HIDDEN LAYER 1   HIDDEN LAYER 2   HIDDEN LAYER 3    OUTPUT LAYER
 *    (14 neurons)        (128 neurons)     (128 neurons)     (64 neurons)     (9 neurons)
 * 
 * ============================================================================
 */

class TableTennisAgent {
    /**
     * Creates a DQN agent for table tennis
     * @param {Object} config - Configuration options
     */
    constructor(config = {}) {
        // Learning parameters
        this.epsilon = config.epsilon || 0.3;
        this.epsilonMin = config.epsilonMin || 0.05;
        this.epsilonDecay = config.epsilonDecay || 0.995;
        this.discountFactor = config.discountFactor || 0.95;
        this.learningRate = config.learningRate || 0.001;
        
        // Training tracking
        this.episodes = 0;
        this.totalSteps = 0;
        this.trainingLoss = 0;
        
        // Experience replay memory
        this.memory = [];
        this.memorySize = config.memorySize || 10000;
        this.batchSize = config.batchSize || 64;
        this.targetUpdateFreq = config.targetUpdateFreq || 100;
        this.trainStep = 0;
        
        // Adaptive strategy tracking
        this.learningStyle = "balanced";
        this.opponentPatterns = {
            leftHits: 0,
            rightHits: 0,
            smashCount: 0,
            totalHits: 0
        };
        
        // Build neural network
        this.model = this._buildNetwork();
        this.targetModel = this._buildNetwork();
        this._updateTargetNetwork();
        
        // Optimizer
        this.optimizer = tf.train.adam(this.learningRate);
        
        console.log('🏓 Table Tennis DQN Agent Initialized');
        console.log('   Network: 14 → 128 → 128 → 64 → 9');
        console.log('   Exploration (ε):', this.epsilon);
        console.log('   Memory Size:', this.memorySize);
    }
    
    /**
     * Build the neural network model
     * @returns {tf.Sequential} TensorFlow.js sequential model
     */
    _buildNetwork() {
        const model = tf.sequential();
        
        model.add(tf.layers.dense({
            units: 128,
            activation: 'relu',
            inputShape: [14]
        }));
        
        model.add(tf.layers.dense({
            units: 128,
            activation: 'relu'
        }));
        
        model.add(tf.layers.dense({
            units: 64,
            activation: 'relu'
        }));
        
        model.add(tf.layers.dense({
            units: 9,
            activation: 'linear'
        }));
        
        return model;
    }
    
    /**
     * Update target network weights from main network
     */
    _updateTargetNetwork() {
        const weights = this.model.getWeights();
        this.targetModel.setWeights(weights);
    }
    
    /**
     * Get observation vector from game state (14 dimensions)
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
        const MAX_POS = 5;
        const MAX_VEL = 15;
        const MAX_TIME = 30;
        const MAX_RALLY = 50;
        
        return [
            ballPos.x / MAX_POS,
            ballPos.y / 3,
            ballPos.z / MAX_POS,
            ballVel.x / MAX_VEL,
            ballVel.y / MAX_VEL,
            ballVel.z / MAX_VEL,
            aiPaddle.x / 2.5,
            aiPaddle.y / 1.5,
            opponentPaddle.x / 2.5,
            predictedLanding.x / MAX_POS,
            predictedLanding.z / MAX_POS,
            Math.min(1, timeToHit / MAX_TIME),
            lastHitType / 8,
            Math.min(1, rallyLength / MAX_RALLY)
        ];
    }
    
    /**
     * Get Q-values for an observation
     * @param {number[]} observation - Array of 14 numbers
     * @returns {Promise<number[]>} Array of 9 Q-values
     */
    async _getQValues(observation) {
        const inputTensor = tf.tensor2d([observation], [1, 14]);
        const qValues = this.model.predict(inputTensor);
        const values = await qValues.data();
        
        inputTensor.dispose();
        qValues.dispose();
        
        return Array.from(values);
    }
    
    /**
     * Select action using epsilon-greedy policy
     * @param {number[]} observation - Current observation
     * @returns {Promise<number>} Action (0-8)
     */
    async getAction(observation) {
        if (Math.random() < this.epsilon) {
            return Math.floor(Math.random() * 9);
        }
        
        const qValues = await this._getQValues(observation);
        return qValues.indexOf(Math.max(...qValues));
    }
    
    /**
     * Calculate reward for an action
     * @param {Object} result - Result of the action
     * @returns {number} Reward value
     */
    calculateReward(result) {
        let reward = 0;
        
        if (result.winRally) reward += 1.0;
        else if (result.loseRally) reward -= 1.0;
        
        if (result.hitMade) reward += 0.1;
        if (result.forcedError) reward += 0.2;
        if (result.unforcedError) reward -= 0.3;
        if (result.hitToCorner) reward += 0.05;
        if (result.usedSpin) reward += 0.05;
        if (result.rallyLength > 5) reward += 0.01;
        
        return reward;
    }
    
    /**
     * Predict where the ball will land
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
     * Train the neural network on a single experience
     * @param {number[]} observation - Current observation
     * @param {number} action - Action taken (0-8)
     * @param {number} reward - Reward received
     * @param {number[]} nextObservation - Next observation
     * @param {boolean} done - Whether episode ended
     */
    async train(observation, action, reward, nextObservation, done) {
        // Store in memory
        this.memory.push({
            observation: observation.slice(),
            action: action,
            reward: reward,
            nextObservation: nextObservation.slice(),
            done: done
        });
        
        if (this.memory.length > this.memorySize) {
            this.memory.shift();
        }
        
        if (this.memory.length < this.batchSize) return;
        
        // Sample random batch
        const batch = [];
        const indices = new Set();
        while (indices.size < this.batchSize) {
            indices.add(Math.floor(Math.random() * this.memory.length));
        }
        for (const idx of indices) batch.push(this.memory[idx]);
        
        const states = batch.map(e => e.observation);
        const nextStates = batch.map(e => e.nextObservation);
        
        // Get current Q-values from main network
        const statesTensor = tf.tensor2d(states, [this.batchSize, 14]);
        const currentQTensor = this.model.predict(statesTensor);
        const currentQ = await currentQTensor.array();
        
        // Get next Q-values from target network
        const nextStatesTensor = tf.tensor2d(nextStates, [this.batchSize, 14]);
        const nextQTensor = this.targetModel.predict(nextStatesTensor);
        const nextQ = await nextQTensor.array();
        
        // Calculate target Q-values using Bellman equation
        const targets = [];
        for (let i = 0; i < this.batchSize; i++) {
            const target = [...currentQ[i]];
            const a = batch[i].action;
            const r = batch[i].reward;
            const d = batch[i].done;
            
            if (d) {
                target[a] = r;
            } else {
                const maxNextQ = Math.max(...nextQ[i]);
                target[a] = r + this.discountFactor * maxNextQ;
            }
            targets.push(target);
        }
        
        // Train the network
        const targetsTensor = tf.tensor2d(targets, [this.batchSize, 9]);
        
        const loss = await this._trainBatch(statesTensor, targetsTensor);
        this.trainingLoss = loss;
        
        // Clean up
        statesTensor.dispose();
        nextStatesTensor.dispose();
        currentQTensor.dispose();
        nextQTensor.dispose();
        targetsTensor.dispose();
        
        // Update target network periodically
        this.trainStep++;
        if (this.trainStep % this.targetUpdateFreq === 0) {
            this._updateTargetNetwork();
        }
        
        // Decay epsilon
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
        
        if (totalReward > 50) {
            this.learningStyle = "confident";
        } else if (totalReward < 10) {
            this.learningStyle = "cautious";
            this.epsilon = Math.min(0.4, this.epsilon + 0.05);
        } else {
            this.learningStyle = "balanced";
        }
        
        console.log(`📊 Episode ${this.episodes} | Reward: ${totalReward.toFixed(1)} | ε: ${this.epsilon.toFixed(3)}`);
    }
    
    /**
     * Update opponent pattern tracking
     * @param {Object} hit - Information about the opponent's hit
     */
    updateOpponentPattern(hit) {
        this.opponentPatterns.totalHits++;
        
        if (hit.targetX < -1) this.opponentPatterns.leftHits++;
        else if (hit.targetX > 1) this.opponentPatterns.rightHits++;
        if (hit.isSmash) this.opponentPatterns.smashCount++;
        
        const leftRatio = this.opponentPatterns.leftHits / this.opponentPatterns.totalHits;
        const rightRatio = this.opponentPatterns.rightHits / this.opponentPatterns.totalHits;
        
        if (leftRatio > 0.6) this.learningStyle = "defendLeft";
        else if (rightRatio > 0.6) this.learningStyle = "defendRight";
        else if (this.opponentPatterns.smashCount > 10) this.learningStyle = "stayBack";
        else this.learningStyle = "balanced";
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
     * Reset the agent
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
     * Save the model
     */
    async saveModel() {
        await this.model.save('downloads://table-tennis-model');
        console.log('💾 Model saved!');
    }
}

// Make available globally
window.TableTennisAgent = TableTennisAgent;

console.log('========================================');
console.log('🏓 Table Tennis RL Module Loaded');
console.log('========================================');
console.log('Class available: TableTennisAgent');
console.log('Use: const agent = new TableTennisAgent();');
console.log('========================================');