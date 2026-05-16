/**
 * ============================================================================
 * TABLE TENNIS - ADAPTIVE REINFORCEMENT LEARNING MODULE
 * ============================================================================
 * 
 * This is Version 3 of our Adaptive RL series.
 * 
 * WHY TABLE TENNIS IS HARDER THAN KING OF THE HILL:
 * ----------------------------------------------------------------------------
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
 * - Continuous action space (paddle position + swing timing)
 * - Ball physics with bounce and spin
 * - Opponent returns ball with different trajectories
 * 
 * ADAPTIVE CHALLENGES IN TABLE TENNIS:
 * ----------------------------------------------------------------------------
 * - Ball speed changes based on hit power
 * - Opponent learns your patterns (where you hit)
 * - Spin adds curve to ball trajectory
 * - Must adapt to different shot types (smash, lob, topspin, backspin)
 * 
 * ============================================================================
 * STATE SPACE (What the agent sees):
 * ----------------------------------------------------------------------------
 * - ballX, ballY, ballZ     : Ball position in 3D space
 * - ballVx, ballVy, ballVz  : Ball velocity (speed and direction)
 * - paddleX, paddleY        : AI paddle position
 * - opponentPaddleX         : Player paddle position (to predict)
 * - ballLandingX, ballLandingZ : Predicted landing position
 * - timeToHit               : Frames until ball reaches paddle
 * - lastHitType             : What shot was last played
 * - rallyLength             : How many hits in current rally
 * 
 * ACTION SPACE (What the agent can do):
 * ----------------------------------------------------------------------------
 * - moveLeft/Right          : Horizontal paddle movement
 * - moveUp/Down             : Vertical paddle movement
 * - hitFlat                 : Standard return (no spin)
 * - hitTopspin              : Ball dips faster after bounce
 * - hitBackspin             : Ball floats and stays low
 * - hitSmash                : Powerful fast shot
 * - hitLob                  : High, deep defensive shot
 * 
 * REWARD STRUCTURE:
 * ----------------------------------------------------------------------------
 * - Win rally               : +1.0
 * - Lose rally              : -1.0
 * - Hit ball over net       : +0.1
 * - Force opponent error    : +0.2
 * - Make unforced error     : -0.3
 * - Hit to corner           : +0.05
 * - Return with spin        : +0.05
 * 
 * ============================================================================
 */

class TableTennisAgent {
    /**
     * Creates a DQN agent for table tennis
     * Neural network: 14 inputs → 128 → 128 → 64 → 7 outputs
     * (Larger network because table tennis is more complex)
     */
    constructor(config = {}) {
        // Learning parameters
        this.epsilon = config.epsilon || 0.3;
        this.episodes = 0;
        this.memory = [];
        this.memorySize = config.memorySize || 5000;
        
        // Table tennis specific parameters
        this.learningStyle = "balanced";  // aggressive, defensive, balanced
        this.weakSpot = "backhand";       // Where opponent misses most
        this.rallyCount = 0;
        
        // Build larger neural network for complex table tennis
        this.model = tf.sequential();
        
        // Input layer: 14 features
        this.model.add(tf.layers.dense({ 
            units: 128, 
            activation: 'relu', 
            inputShape: [14] 
        }));
        
        // Hidden layer 2
        this.model.add(tf.layers.dense({ 
            units: 128, 
            activation: 'relu' 
        }));
        
        // Hidden layer 3
        this.model.add(tf.layers.dense({ 
            units: 64, 
            activation: 'relu' 
        }));
        
        // Output layer: 7 actions
        this.model.add(tf.layers.dense({ 
            units: 7, 
            activation: 'linear' 
        }));
        
        this.model.compile({ 
            optimizer: 'adam', 
            loss: 'meanSquaredError' 
        });
        
        console.log('🏓 Table Tennis DQN Agent Initialized');
        console.log('   Network: 14 → 128 → 128 → 64 → 7');
        console.log('   Exploration rate (ε):', this.epsilon);
        console.log('   Memory size:', this.memorySize);
    }
    
    /**
     * Get observation from game state
     * 14 features for table tennis:
     * 0-2: Ball position (x, y, z)
     * 3-5: Ball velocity (vx, vy, vz)
     * 6-7: AI paddle position (x, y)
     * 8:   Opponent paddle X position
     * 9-10: Predicted landing position
     * 11:  Time to hit (frames)
     * 12:  Last hit type (encoded)
     * 13:  Rally length (normalized)
     */
    getObservation(ball, ballVel, aiPaddle, opponentPaddle, predictedLanding, timeToHit, lastHitType, rallyLength) {
        return [
            ball.x / 5, ball.y / 3, ball.z / 10,           // Ball position (normalized)
            ballVel.x / 20, ballVel.y / 15, ballVel.z / 20, // Ball velocity
            aiPaddle.x / 2, aiPaddle.y / 2,                 // AI paddle position
            opponentPaddle.x / 2,                           // Opponent paddle X
            predictedLanding.x / 5, predictedLanding.z / 5, // Landing prediction
            Math.min(1, timeToHit / 30),                    // Time to hit (normalized)
            lastHitType / 6,                                // Hit type (0-1)
            Math.min(1, rallyLength / 20)                   // Rally length
        ];
    }
    
    /**
     * Get action using epsilon-greedy
     * Actions: 0=moveLeft, 1=moveRight, 2=moveUp, 3=moveDown, 
     *          4=hitFlat, 5=hitTopspin, 6=hitBackspin, 7=hitSmash, 8=hitLob
     */
    async getAction(observation) {
        const input = tf.tensor2d([observation], [1, 14]);
        const qValues = this.model.predict(input);
        const values = await qValues.data();
        
        input.dispose();
        qValues.dispose();
        
        if (Math.random() < this.epsilon) {
            return Math.floor(Math.random() * 9);  // 9 actions
        }
        return values.indexOf(Math.max(...values));
    }
    
    /**
     * Train on experience
     */
    async train(obs, action, reward, nextObs, done) {
        this.memory.push({ obs, action, reward, nextObs, done });
        if (this.memory.length > this.memorySize) this.memory.shift();
        if (this.memory.length < 64) return;  // Need larger batch for complex game
        
        // Sample batch of 64 experiences
        const batch = [];
        const indices = new Set();
        while (indices.size < 64) {
            indices.add(Math.floor(Math.random() * this.memory.length));
        }
        for (const idx of indices) batch.push(this.memory[idx]);
        
        const states = batch.map(e => e.obs);
        const nextStates = batch.map(e => e.nextObs);
        
        const currentQTensor = this.model.predict(tf.tensor2d(states, [64, 14]));
        const currentQ = await currentQTensor.array();
        const nextQTensor = this.model.predict(tf.tensor2d(nextStates, [64, 14]));
        const nextQ = await nextQTensor.array();
        
        const targets = [];
        for (let i = 0; i < 64; i++) {
            const target = [...currentQ[i]];
            const a = batch[i].action;
            const r = batch[i].reward;
            const d = batch[i].done;
            
            if (d) {
                target[a] = r;
            } else {
                target[a] = r + 0.95 * Math.max(...nextQ[i]);
            }
            targets.push(target);
        }
        
        const xs = tf.tensor2d(states, [64, 14]);
        const ys = tf.tensor2d(targets, [64, 9]);
        await this.model.fit(xs, ys, { epochs: 1, verbose: 0 });
        
        xs.dispose();
        ys.dispose();
        currentQTensor.dispose();
        nextQTensor.dispose();
        
        this.epsilon = Math.max(0.05, this.epsilon * 0.995);
    }
    
    /**
     * Adaptive strategy based on opponent patterns
     */
    adaptStrategy(opponentPattern) {
        if (opponentPattern.hitsToLeft > 0.6) {
            this.weakSpot = "left";
            this.learningStyle = "aggressive";
        } else if (opponentPattern.hitsToRight > 0.6) {
            this.weakSpot = "right";
            this.learningStyle = "aggressive";
        } else if (opponentPattern.smashCount > 10) {
            this.learningStyle = "defensive";
            this.weakSpot = "deep";
        } else {
            this.learningStyle = "balanced";
        }
    }
    
    incrementEpisode() {
        this.episodes++;
        this.updateUI();
    }
    
    updateUI() {
        const epsilonEl = document.getElementById('epsilon');
        const episodesEl = document.getElementById('train-episodes');
        const memoryEl = document.getElementById('memory');
        const styleEl = document.getElementById('ai-style');
        
        if (epsilonEl) epsilonEl.textContent = this.epsilon.toFixed(3);
        if (episodesEl) episodesEl.textContent = this.episodes;
        if (memoryEl) memoryEl.textContent = this.memory.length;
        if (styleEl) styleEl.textContent = this.learningStyle;
    }
    
    getEpsilon() {
        return this.epsilon;
    }
}

window.TableTennisAgent = TableTennisAgent;
