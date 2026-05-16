/**
 * Adaptive Reinforcement Learning Module
 * For teaching adaptive RL concepts - King of the Hill environment
 * 
 * Features:
 * - Dynamic goal repositioning (hill moves randomly)
 * - Adaptive reward shaping based on goal change timing
 * - Exploration vs exploitation with adaptive epsilon
 * - Q-learning with experience replay
 * - Performance tracking and metrics
 * 
 * @author Evans Enonchong
 * @version 2.0
 */

// ============================================
// 1. ENVIRONMENT CLASS
// ============================================

class AdaptiveHillEnvironment {
    /**
     * The environment that the agent interacts with
     * Handles state updates, rewards, and goal repositioning
     */
    constructor(config = {}) {
        // World parameters
        this.arenaSize = config.arenaSize || 20;
        this.hillRadius = config.hillRadius || 2.5;
        this.hillMoveInterval = config.hillMoveInterval || 100;  // Steps between moves
        this.agentSpeed = config.agentSpeed || 0.5;
        
        // Adaptive parameters (these change over time!)
        this.adaptiveInterval = config.adaptiveInterval !== false;
        this.adaptiveIntervalRange = config.adaptiveIntervalRange || { min: 40, max: 160 };
        
        // Internal state
        this.agentPos = { x: 0, z: 0 };
        this.goalPos = { x: 0, z: 0 };
        this.stepCount = 0;
        this.stepsSinceLastMove = 0;
        this.currentInterval = this.hillMoveInterval;
        this.episodeReward = 0;
        this.totalReward = 0;
        
        // Tracking for adaptive metrics
        this.goalChanges = [];
        this.adaptationTimes = [];
        this.lastZoneEntryStep = 0;
        
        // Performance tracking
        this.episodeCount = 0;
        this.recentRewards = [];
        
        // Initialize
        this.reset();
    }
    
    /**
     * Reset environment for new episode
     * @returns {Object} Initial observation
     */
    reset() {
        // Random starting position (avoid center bias)
        this.agentPos = {
            x: (Math.random() - 0.5) * (this.arenaSize - 6),
            z: (Math.random() - 0.5) * (this.arenaSize - 6)
        };
        
        // Random initial goal
        this.goalPos = this._getRandomPosition();
        
        // Reset counters
        this.stepCount = 0;
        this.stepsSinceLastMove = 0;
        this.episodeReward = 0;
        
        // Adapt the environment! (key adaptive feature)
        if (this.adaptiveInterval && this.episodeCount > 0) {
            this._adaptInterval();
        }
        
        this.episodeCount++;
        
        return this._getObservation();
    }
    
    /**
     * Take an action and get next state, reward, and done flag
     * @param {number} action - 0:Up,1:Down,2:Left,3:Right,4:Stay
     * @returns {Object} {observation, reward, done, info}
     */
    step(action) {
        // Action mapping
        const moves = {
            0: { x: 0, z: 1 },   // Up
            1: { x: 0, z: -1 },  // Down
            2: { x: -1, z: 0 },  // Left
            3: { x: 1, z: 0 },   // Right
            4: { x: 0, z: 0 }    // Stay
        };
        
        const move = moves[action];
        
        // Apply movement with boundary constraints
        let newX = this.agentPos.x + move.x * this.agentSpeed;
        let newZ = this.agentPos.z + move.z * this.agentSpeed;
        
        newX = Math.max(-this.arenaSize + 2, Math.min(this.arenaSize - 2, newX));
        newZ = Math.max(-this.arenaSize + 2, Math.min(this.arenaSize - 2, newZ));
        
        this.agentPos = { x: newX, z: newZ };
        
        // Calculate distance to goal
        const distance = this._getDistance();
        const inZone = distance < this.hillRadius;
        
        // Calculate reward (ADAPTIVE reward shaping!)
        let reward = this._calculateReward(distance, inZone);
        
        // Check if goal should move (dynamic repositioning)
        this.stepsSinceLastMove++;
        let goalMoved = false;
        
        if (this.stepsSinceLastMove >= this.currentInterval) {
            this.goalPos = this._getRandomPosition();
            this.stepsSinceLastMove = 0;
            goalMoved = true;
            
            // Track goal change for metrics
            this.goalChanges.push({
                step: this.stepCount,
                position: { ...this.goalPos }
            });
        }
        
        // Track adaptation speed (how quickly agent re-enters zone after move)
        if (goalMoved) {
            const stepsSinceLastZone = this.stepCount - this.lastZoneEntryStep;
            this.adaptationTimes.push(stepsSinceLastZone);
        }
        
        // Track when agent is in zone
        if (inZone) {
            this.lastZoneEntryStep = this.stepCount;
        }
        
        this.stepCount++;
        this.episodeReward += reward;
        this.totalReward += reward;
        
        // Episode ends after 1000 steps
        const done = this.stepCount >= 1000;
        
        return {
            observation: this._getObservation(),
            reward: reward,
            done: done,
            info: {
                distance: distance,
                inZone: inZone,
                goalMoved: goalMoved,
                episodeReward: this.episodeReward,
                stepsRemaining: 1000 - this.stepCount,
                currentInterval: this.currentInterval
            }
        };
    }
    
    /**
     * ADAPTIVE REWARD SHAPING
     * Changes based on timing since goal moved
     */
    _calculateReward(distance, inZone) {
        let reward = 0;
        
        // Primary reward: being in the hill zone
        if (inZone) {
            reward += 1.0;  // +1 per step in zone
        }
        
        // Small time penalty to encourage efficiency
        reward -= 0.01;
        
        // ADAPTIVE shaping: early after goal move, reward speed
        if (this.stepsSinceLastMove < 20) {
            // Encourage reaching the new hill FAST
            const speedBonus = Math.max(0, (1 - distance / this.arenaSize) * 0.5);
            reward += speedBonus;
        } else {
            // Later, reward staying in the hill
            if (inZone) {
                reward += 0.1;  // Staying bonus
            }
        }
        
        return reward;
    }
    
    /**
     * ADAPTIVE INTERVAL ADJUSTMENT
     * The environment changes its dynamics over time!
     * This is the key adaptive feature
     */
    _adaptInterval() {
        const oldInterval = this.currentInterval;
        
        // Multiple adaptation strategies
        const strategies = ['random', 'faster', 'slower', 'cyclic'];
        const strategy = strategies[Math.floor(Math.random() * strategies.length)];
        
        switch(strategy) {
            case 'random':
                // Completely random interval
                this.currentInterval = Math.floor(
                    this.adaptiveIntervalRange.min + 
                    Math.random() * (this.adaptiveIntervalRange.max - this.adaptiveIntervalRange.min)
                );
                break;
                
            case 'faster':
                // Make goal move MORE frequently (harder!)
                this.currentInterval = Math.max(
                    this.adaptiveIntervalRange.min,
                    this.currentInterval * 0.7
                );
                break;
                
            case 'slower':
                // Make goal move LESS frequently (easier)
                this.currentInterval = Math.min(
                    this.adaptiveIntervalRange.max,
                    this.currentInterval * 1.5
                );
                break;
                
            case 'cyclic':
                // Cyclic pattern - tests if agent can detect rhythms
                const cycle = Math.sin(this.episodeCount * 0.1);
                const range = this.adaptiveIntervalRange.max - this.adaptiveIntervalRange.min;
                this.currentInterval = Math.floor(
                    this.adaptiveIntervalRange.min + ((cycle + 1) / 2) * range
                );
                break;
        }
        
        console.log(`🔄 ENVIRONMENT ADAPTED: Goal move interval ${oldInterval} → ${this.currentInterval} steps`);
    }
    
    /**
     * Get random position within arena
     */
    _getRandomPosition() {
        const margin = 3;
        return {
            x: (Math.random() - 0.5) * (this.arenaSize - margin * 2),
            z: (Math.random() - 0.5) * (this.arenaSize - margin * 2)
        };
    }
    
    /**
     * Calculate Euclidean distance to goal
     */
    _getDistance() {
        const dx = this.agentPos.x - this.goalPos.x;
        const dz = this.agentPos.z - this.goalPos.z;
        return Math.sqrt(dx * dx + dz * dz);
    }
    
    /**
     * Get normalized observation vector
     * @returns {Float32Array} [agent_x, agent_z, goal_x, goal_z, time_norm, in_zone_flag]
     */
    _getObservation() {
        const normX = this.agentPos.x / this.arenaSize;
        const normZ = this.agentPos.z / this.arenaSize;
        const goalNormX = this.goalPos.x / this.arenaSize;
        const goalNormZ = this.goalPos.z / this.arenaSize;
        const timeNorm = this.stepsSinceLastMove / this.currentInterval;
        const inZone = this._getDistance() < this.hillRadius ? 1 : 0;
        
        return new Float32Array([normX, normZ, goalNormX, goalNormZ, timeNorm, inZone]);
    }
    
    /**
     * Get performance metrics
     */
    getMetrics() {
        const avgAdaptation = this.adaptationTimes.length > 0
            ? this.adaptationTimes.reduce((a,b) => a + b, 0) / this.adaptationTimes.length
            : null;
        
        return {
            episode: this.episodeCount,
            currentInterval: this.currentInterval,
            totalGoalChanges: this.goalChanges.length,
            averageAdaptationTime: avgAdaptation,
            episodeReward: this.episodeReward,
            totalReward: this.totalReward
        };
    }
}


// ============================================
// 2. ADAPTIVE DQN AGENT
// ============================================

class AdaptiveDQNAgent {
    /**
     * Q-Learning agent that adapts to changing environment
     * Features:
     * - Adaptive exploration rate
     * - Experience replay
     * - Performance-based learning rate adjustment
     */
    constructor(config = {}) {
        // Learning parameters
        this.learningRate = config.learningRate || 0.1;
        this.discountFactor = config.discountFactor || 0.95;
        this.epsilon = config.epsilon || 0.3;  // Initial exploration rate
        this.epsilonMin = config.epsilonMin || 0.05;
        this.epsilonDecay = config.epsilonDecay || 0.995;
        
        // Memory for experience replay
        this.memory = [];
        this.memorySize = config.memorySize || 5000;
        
        // Q-table: maps state-action pairs to values
        this.qTable = new Map();
        
        // Tracking for adaptation
        this.recentRewards = [];
        this.episodeRewards = [];
        this.totalSteps = 0;
        this.goalChangeDetected = false;
        
        // Performance tracking
        this.lastAverageReward = 0;
    }
    
    /**
     * Select action using epsilon-greedy policy
     * @param {Float32Array} observation - Current state
     * @param {Object} info - Additional info (like goalMoved flag)
     * @returns {number} Action (0-4)
     */
    act(observation, info = {}) {
        // ADAPTIVE BEHAVIOR: When goal moves, increase exploration!
        if (info.goalMoved) {
            this.goalChangeDetected = true;
            // Temporarily boost exploration to find new goal
            const boostedEpsilon = Math.min(0.5, this.epsilon + 0.2);
            if (Math.random() < boostedEpsilon) {
                return Math.floor(Math.random() * 5);
            }
        } else {
            this.goalChangeDetected = false;
        }
        
        // Normal epsilon-greedy
        if (Math.random() < this.epsilon) {
            // Explore: random action
            return Math.floor(Math.random() * 5);
        } else {
            // Exploit: best known action
            return this._getBestAction(observation);
        }
    }
    
    /**
     * Get best action for given state
     */
    _getBestAction(observation) {
        const stateKey = this._getStateKey(observation);
        let bestAction = 0;
        let bestValue = -Infinity;
        
        for (let a = 0; a < 5; a++) {
            const value = this._getQValue(stateKey, a);
            if (value > bestValue) {
                bestValue = value;
                bestAction = a;
            }
        }
        
        return bestAction;
    }
    
    /**
     * Train the agent on a transition
     * @param {Float32Array} observation - Previous state
     * @param {number} action - Action taken
     * @param {number} reward - Reward received
     * @param {Float32Array} nextObservation - Next state
     * @param {boolean} done - Episode ended?
     * @param {Object} info - Additional info
     */
    train(observation, action, reward, nextObservation, done, info = {}) {
        // Store transition in memory
        this.memory.push({
            observation: observation.slice(),
            action: action,
            reward: reward,
            nextObservation: nextObservation.slice(),
            done: done
        });
        
        // Limit memory size
        if (this.memory.length > this.memorySize) {
            this.memory.shift();
        }
        
        // Perform Q-learning update
        this._qLearningUpdate(observation, action, reward, nextObservation, done);
        
        // Track recent rewards for adaptive learning
        this.recentRewards.push(reward);
        if (this.recentRewards.length > 100) {
            this.recentRewards.shift();
        }
        
        // ADAPTIVE EPSILON DECAY based on performance
        this._adaptiveEpsilonDecay();
        
        // ADAPTIVE LEARNING RATE adjustment
        if (this.totalSteps % 500 === 0) {
            this._adjustLearningRate();
        }
        
        this.totalSteps++;
    }
    
    /**
     * Core Q-learning update
     */
    _qLearningUpdate(observation, action, reward, nextObservation, done) {
        const stateKey = this._getStateKey(observation);
        const nextStateKey = this._getStateKey(nextObservation);
        
        // Current Q-value
        const currentQ = this._getQValue(stateKey, action);
        
        // Target Q-value (Bellman equation)
        let maxNextQ = 0;
        if (!done) {
            for (let a = 0; a < 5; a++) {
                maxNextQ = Math.max(maxNextQ, this._getQValue(nextStateKey, a));
            }
        }
        
        const targetQ = reward + this.discountFactor * maxNextQ;
        
        // Update Q-value
        const newQ = currentQ + this.learningRate * (targetQ - currentQ);
        this._setQValue(stateKey, action, newQ);
    }
    
    /**
     * ADAPTIVE EPSILON DECAY
     * Explores more when performing poorly
     */
    _adaptiveEpsilonDecay() {
        if (this.recentRewards.length < 50) {
            this.epsilon *= this.epsilonDecay;
        } else {
            const avgReward = this.recentRewards.reduce((a,b) => a + b, 0) / 50;
            
            if (avgReward > 0.3) {
                // Doing well: decay faster (exploit more)
                this.epsilon *= 0.99;
            } else if (avgReward < 0.1) {
                // Doing poorly: decay slower (explore more)
                this.epsilon *= 0.998;
            } else {
                this.epsilon *= this.epsilonDecay;
            }
        }
        
        this.epsilon = Math.max(this.epsilonMin, this.epsilon);
    }
    
    /**
     * ADAPTIVE LEARNING RATE
     * Adjusts based on performance trends
     */
    _adjustLearningRate() {
        if (this.episodeRewards.length < 10) return;
        
        const last5 = this.episodeRewards.slice(-5).reduce((a,b) => a + b, 0) / 5;
        const prev5 = this.episodeRewards.slice(-10, -5).reduce((a,b) => a + b, 0) / 5;
        
        if (last5 < prev5 * 0.8) {
            // Performance dropping: increase learning rate
            this.learningRate = Math.min(0.3, this.learningRate * 1.1);
            console.log(`📈 Learning rate increased to ${this.learningRate.toFixed(3)}`);
        } else if (last5 > prev5 * 1.2 && this.learningRate > 0.05) {
            // Performance improving: decrease learning rate
            this.learningRate *= 0.95;
        }
    }
    
    /**
     * Called at end of episode
     */
    endEpisode(totalReward) {
        this.episodeRewards.push(totalReward);
        
        // Keep only last 100 episodes
        if (this.episodeRewards.length > 100) {
            this.episodeRewards.shift();
        }
        
        // Boost epsilon if recent performance is poor
        const avgRecent = this.episodeRewards.slice(-5).reduce((a,b) => a + b, 0) / 5;
        if (avgRecent < 10 && this.epsilon < 0.2) {
            this.epsilon = 0.3;
            console.log(`🔄 Exploration boosted due to poor performance`);
        }
    }
    
    /**
     * Convert continuous observation to discrete state key
     * Discretizes the 6-dimensional observation into bins
     */
    _getStateKey(observation) {
        const bins = 8;  // 8 bins per dimension = 8^6 = 262k possible states
        const keys = [];
        
        for (let i = 0; i < observation.length; i++) {
            // Map from [-1, 1] to [0, bins-1]
            const bin = Math.floor((observation[i] + 1) / 2 * bins);
            keys.push(Math.min(bins - 1, Math.max(0, bin)));
        }
        
        return keys.join(',');
    }
    
    /**
     * Get Q-value from table
     */
    _getQValue(stateKey, action) {
        const key = `${stateKey}|${action}`;
        return this.qTable.get(key) || 0;
    }
    
    /**
     * Set Q-value in table
     */
    _setQValue(stateKey, action, value) {
        const key = `${stateKey}|${action}`;
        this.qTable.set(key, value);
    }
    
    /**
     * Get agent metrics for visualization
     */
    getMetrics() {
        const avgRecentReward = this.recentRewards.length > 0
            ? this.recentRewards.reduce((a,b) => a + b, 0) / this.recentRewards.length
            : 0;
        
        const avgEpisodeReward = this.episodeRewards.length > 0
            ? this.episodeRewards.reduce((a,b) => a + b, 0) / this.episodeRewards.length
            : 0;
        
        return {
            epsilon: this.epsilon.toFixed(3),
            learningRate: this.learningRate.toFixed(3),
            memorySize: this.memory.length,
            totalSteps: this.totalSteps,
            averageReward: avgRecentReward.toFixed(2),
            averageEpisodeReward: avgEpisodeReward.toFixed(2),
            qTableSize: this.qTable.size
        };
    }
}


// ============================================
// 3. TRAINING LOOP
// ============================================

class AdaptiveRLTrainer {
    /**
     * Coordinates training between environment and agent
     * Tracks metrics and provides visualization data
     */
    constructor(env, agent) {
        this.env = env;
        this.agent = agent;
        this.trainingActive = false;
        this.currentEpisode = 0;
        this.episodeData = [];
        this.listeners = [];
        
        // Performance tracking
        this.rewardHistory = [];
        this.adaptationHistory = [];
    }
    
    /**
     * Add event listener for training updates
     */
    onUpdate(callback) {
        this.listeners.push(callback);
    }
    
    /**
     * Notify all listeners of update
     */
    _notify(data) {
        this.listeners.forEach(cb => cb(data));
    }
    
    /**
     * Train for specified number of episodes
     */
    async train(numEpisodes, stepsPerEpisode = 500) {
        this.trainingActive = true;
        
        for (let ep = 0; ep < numEpisodes && this.trainingActive; ep++) {
            const episodeResult = await this._runEpisode(stepsPerEpisode);
            this.rewardHistory.push(episodeResult);
            
            // Notify listeners for real-time updates
            this._notify({
                type: 'episode_complete',
                episode: this.currentEpisode,
                reward: episodeResult.totalReward,
                metrics: this.env.getMetrics(),
                agentMetrics: this.agent.getMetrics(),
                adaptationTime: episodeResult.adaptationTime
            });
            
            // Small delay to prevent UI freezing
            await this._sleep(10);
        }
        
        this.trainingActive = false;
        return this.rewardHistory;
    }
    
    /**
     * Run a single episode
     */
    async _runEpisode(maxSteps) {
        let observation = this.env.reset();
        let done = false;
        let step = 0;
        let totalReward = 0;
        let adaptationTimes = [];
        
        while (!done && step < maxSteps) {
            // Get action from agent
            const action = this.agent.act(observation, { goalMoved: false });
            
            // Take step in environment
            const result = this.env.step(action);
            
            // Train agent
            this.agent.train(
                observation, 
                action, 
                result.reward, 
                result.observation, 
                result.done,
                { goalMoved: result.info.goalMoved }
            );
            
            totalReward += result.reward;
            
            // Track adaptation when goal moves
            if (result.info.goalMoved) {
                adaptationTimes.push(result.info.stepsRemaining);
            }
            
            // Send step update for visualization
            this._notify({
                type: 'step',
                episode: this.currentEpisode + 1,
                step: step,
                observation: result.observation,
                reward: result.reward,
                inZone: result.info.inZone,
                distance: result.info.distance,
                goalMoved: result.info.goalMoved,
                totalReward: totalReward,
                agentMetrics: this.agent.getMetrics(),
                envMetrics: this.env.getMetrics()
            });
            
            observation = result.observation;
            done = result.done;
            step++;
        }
        
        this.agent.endEpisode(totalReward);
        this.currentEpisode++;
        
        const avgAdaptation = adaptationTimes.length > 0
            ? adaptationTimes.reduce((a,b) => a + b, 0) / adaptationTimes.length
            : null;
        
        return {
            episode: this.currentEpisode,
            totalReward: totalReward,
            steps: step,
            adaptationTime: avgAdaptation,
            goalChanges: this.env.goalChanges.length
        };
    }
    
    /**
     * Stop training
     */
    stop() {
        this.trainingActive = false;
    }
    
    /**
     * Sleep helper
     */
    _sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
    
    /**
     * Get summary statistics
     */
    getSummary() {
        const avgReward = this.rewardHistory.reduce((a,b) => a + b.totalReward, 0) / this.rewardHistory.length;
        const bestReward = Math.max(...this.rewardHistory.map(r => r.totalReward));
        
        return {
            episodesCompleted: this.currentEpisode,
            averageReward: avgReward,
            bestReward: bestReward,
            finalEpsilon: this.agent.epsilon,
            finalQTableSize: this.agent.qTable.size
        };
    }
}


// ============================================
// 4. EXPORT FOR USE IN HTML
// ============================================

// Make available globally
window.AdaptiveRL = {
    Environment: AdaptiveHillEnvironment,
    Agent: AdaptiveDQNAgent,
    Trainer: AdaptiveRLTrainer
};

console.log('Adaptive RL Module Loaded!');
console.log('Available classes:');
console.log('  - AdaptiveRL.Environment');
console.log('  - AdaptiveRL.Agent');
console.log('  - AdaptiveRL.Trainer');