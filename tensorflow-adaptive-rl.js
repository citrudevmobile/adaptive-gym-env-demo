/**
 * TENSORFLOW.JS ADAPTIVE REINFORCEMENT LEARNING MODULE
 * =====================================================
 * 
 * A complete educational implementation of Adaptive Reinforcement Learning
 * using TensorFlow.js Deep Q-Networks (DQN) in pure browser JavaScript.
 * 
 * ============================================================================
 * HOW TO USE THIS MODULE
 * ============================================================================
 * 
 * STEP 1: Include TensorFlow.js CDN in your HTML:
 * -------------------------------------------------
 * <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.15.0/dist/tf.min.js"></script>
 * 
 * STEP 2: Include this file:
 * -------------------------------------------------
 * <script src="tensorflow-adaptive-rl.js"></script>
 * 
 * STEP 3: Use the classes in your code:
 * -------------------------------------------------
 * let env = new TensorFlowRL.Environment();
 * let agent = new TensorFlowRL.DQNAgent();
 * let trainer = new TensorFlowRL.Trainer(env, agent);
 * trainer.train(100);  // Train for 100 episodes
 * 
 * ============================================================================
 * WHAT IS ADAPTIVE REINFORCEMENT LEARNING?
 * ============================================================================
 * 
 * Normal Reinforcement Learning:
 * -------------------------------
 * The environment rules never change. The agent learns ONE strategy and uses it forever.
 * Example: A chess AI that always plays the same opening.
 * 
 * Adaptive Reinforcement Learning:
 * --------------------------------
 * The environment rules CHANGE over time. The agent must KEEP LEARNING and ADAPTING.
 * Example: A robot in a factory where machinery keeps getting moved around.
 * 
 * In this King of the Hill game, the "hill" (golden zone) moves to random positions,
 * and the SPEED at which it moves changes over time. The agent must adapt!
 * 
 * ============================================================================
 * WHAT IS A DEEP Q-NETWORK (DQN)?
 * ============================================================================
 * 
 * Q-Table (Tabular Q-Learning):
 * -----------------------------
 * Uses a big lookup table: for every possible state, store values for each action.
 * Problem: What if there are millions of possible states? The table gets too big!
 * 
 * Deep Q-Network (DQN):
 * ---------------------
 * Uses a NEURAL NETWORK to approximate the Q-values instead of storing them all.
 * The neural network learns to PREDICT the Q-value for any state, even ones it's never seen!
 * 
 * Think of it this way:
 * - Q-Table = memorizing the answer to every possible math problem
 * - DQN = learning the FORMULA to solve any math problem
 * 
 * ============================================================================
 * THE NEURAL NETWORK ARCHITECTURE
 * ============================================================================
 * 
 *                    INPUT LAYER
 *                   (6 neurons)
 *                        |
 *                        ▼
 *             HIDDEN LAYER 1
 *                 (64 neurons)
 *                  ReLU activation
 *                        |
 *                        ▼
 *             HIDDEN LAYER 2
 *                 (64 neurons)
 *                  ReLU activation
 *                        |
 *                        ▼
 *                   OUTPUT LAYER
 *                   (5 neurons)
 *                  Linear activation
 * 
 * WHAT EACH LAYER DOES:
 * ---------------------
 * 
 * INPUT LAYER (6 neurons):
 *   Neuron 0: Agent's X position (normalized -1 to 1)
 *   Neuron 1: Agent's Z position (normalized -1 to 1)
 *   Neuron 2: Hill's X position (normalized -1 to 1)
 *   Neuron 3: Hill's Z position (normalized -1 to 1)
 *   Neuron 4: Time until hill moves (0 to 1)
 *   Neuron 5: Is agent in hill? (0 or 1)
 * 
 * HIDDEN LAYERS (64 neurons each):
 *   These layers learn PATTERNS in the data.
 *   Examples of patterns they might learn:
 *   - "If I'm far from the hill, I should move toward it"
 *   - "If the hill is about to move, I should go to the center"
 *   - "If I'm in the hill, I should stay still"
 * 
 *   ReLU Activation: f(x) = max(0, x)
 *   This means negative values become 0. This helps the network learn
 *   non-linear relationships (patterns that aren't straight lines).
 * 
 * OUTPUT LAYER (5 neurons):
 *   Each neuron outputs a Q-value for one action:
 *   Neuron 0: How good is moving UP?
 *   Neuron 1: How good is moving DOWN?
 *   Neuron 2: How good is moving LEFT?
 *   Neuron 3: How good is moving RIGHT?
 *   Neuron 4: How good is STAYING?
 * 
 * ============================================================================
 * KEY DQN CONCEPTS EXPLAINED
 * ============================================================================
 * 
 * 1. EXPERIENCE REPLAY:
 * --------------------
 * Instead of learning from each experience once and forgetting it,
 * we store experiences in a "memory" and learn from random batches.
 * This is like studying for a test by reviewing old homework.
 * 
 * Why is this important?
 * - Prevents the network from forgetting past lessons
 * - Breaks correlations between consecutive experiences
 * - Makes learning more stable
 * 
 * 2. TARGET NETWORK:
 * -----------------
 * We use TWO neural networks:
 * - Main network: Learns and updates every step
 * - Target network: Used to calculate targets, updates slowly
 * 
 * Why two networks?
 * If you use the same network to both predict and calculate targets,
 * you create a feedback loop. The network chases its own predictions.
 * Using a separate target network stabilizes training.
 * 
 * 3. BELLMAN EQUATION:
 * -------------------
 * The formula that updates Q-values:
 * 
 * Q(s,a) = Q(s,a) + α * [ R + γ * max(Q(s',a')) - Q(s,a) ]
 * 
 * Where:
 * - Q(s,a) = Current estimate for taking action 'a' in state 's'
 * - α (alpha) = Learning rate (how much to trust new information)
 * - R = Immediate reward received
 * - γ (gamma) = Discount factor (how much to value future rewards)
 * - max(Q(s',a')) = Best possible future reward from new state
 * 
 * 4. EPSILON-GREEDY EXPLORATION:
 * -----------------------------
 * With probability ε (epsilon): Take RANDOM action (explore)
 * With probability 1-ε: Take BEST action according to network (exploit)
 * 
 * This balances trying new things vs using what you know.
 * 
 * ============================================================================
 * THE ADAPTIVE FEATURES
 * ============================================================================
 * 
 * Feature 1: Environment Adaptation
 * ---------------------------------
 * The speed at which the hill moves changes over time!
 * Possible changes: faster, slower, random, or cyclic.
 * This forces the agent to keep adapting.
 * 
 * Feature 2: Adaptive Exploration
 * -------------------------------
 * When the agent is doing poorly, epsilon decays slower (explore more).
 * When the agent is doing well, epsilon decays faster (exploit more).
 * 
 * Feature 3: Goal Change Response
 * -------------------------------
 * When the hill moves, we temporarily INCREASE exploration.
 * Why? Because old strategies might not work at the new location.
 * 
 * ============================================================================
 * THE THREE CLASSES
 * ============================================================================
 * 
 * 1. AdaptiveHillEnvironment - The game world
 *    - Defines the rules
 *    - Tracks agent and hill positions
 *    - Calculates rewards
 *    - Moves the hill at random intervals
 * 
 * 2. TensorFlowDQNAgent - The learning AI
 *    - Builds and manages the neural network
 *    - Selects actions using epsilon-greedy
 *    - Learns from experiences using Q-learning
 *    - Uses experience replay and target network
 * 
 * 3. AdaptiveRLTrainer - The training coordinator
 *    - Runs episodes
 *    - Calls environment and agent at the right times
 *    - Broadcasts updates for visualization
 * 
 * @author Adaptive RL Tutorial
 * @version 3.0 - TensorFlow.js Deep Q-Network Edition
 */

// ============================================================================
// WAIT FOR TENSORFLOW TO LOAD
// ============================================================================

(function() {
    
    // Check if TensorFlow.js is loaded
    if (typeof tf === 'undefined') {
        console.error('ERROR: TensorFlow.js not loaded!');
        console.error('Add this script tag to your HTML:');
        console.error('<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.15.0/dist/tf.min.js"></script>');
        return;
    }
    
    console.log('TensorFlow.js version:', tf.version.tfjs);
    console.log('Loading Adaptive RL Module...');

// ============================================================================
// PART 1: THE ENVIRONMENT CLASS
// ============================================================================
//
// WHAT IS THE ENVIRONMENT?
// ------------------------
// The environment is the GAME WORLD. It contains all the rules:
// - Where is the agent? (blue character)
// - Where is the hill? (golden zone)
// - What happens when the agent moves?
// - How many points does the agent get?
// - When does the hill move?
//
// THINK OF IT LIKE THIS:
// ---------------------
// If you were playing a game of tag, the environment is:
// - The playground (boundaries)
// - Where the "safe zone" is (the hill)
// - Where you are standing
// - The rules about scoring
// - The points you earn for reaching safe zone
//
// THE ENVIRONMENT'S THREE MAIN JOBS:
// ---------------------------------
// 1. reset() - Start a brand new game
// 2. step(action) - Let the agent take one move, then tell them what happened
// 3. _getObservation() - Tell the agent what they can "see" (their senses)
//

class AdaptiveHillEnvironment {
    
    // ------------------------------------------------------------------------
    // CONSTRUCTOR - SETTING UP THE GAME WORLD
    // ------------------------------------------------------------------------
    // When you create a new environment, this function runs automatically.
    // It sets up all the rules and starting values.
    //
    // EXAMPLE: let game = new AdaptiveHillEnvironment({ arenaSize: 30 })
    // This creates a bigger arena (30 units wide instead of 20)
    //
    // QUESTION FOR YOU: What happens if you don't pass any configuration?
    // ANSWER: It uses the default values (arenaSize=20, hillRadius=2.5, etc.)
    //
    
    constructor(config = {}) {
        
        // ----- WORLD SIZE AND SHAPE -----
        // These are like the "rules of the playground"
        
        // arenaSize: How big is the playing field?
        // The arena is a square. arenaSize = 20 means 20 units wide and 20 units deep.
        // The agent can move from -18 to +18 in both directions.
        // Default value is 20 if not specified.
        this.arenaSize = config.arenaSize || 20;
        
        // hillRadius: How big is the scoring zone?
        // When the agent gets within this distance of the hill center, they are "in the zone"
        // A radius of 2.5 means the hill is a circle that is 5 units across.
        // If you make this larger, the game becomes easier (bigger target).
        this.hillRadius = config.hillRadius || 2.5;
        
        // hillMoveInterval: How many steps until the hill moves?
        // After this many actions, the hill teleports to a random new location.
        // Default is 100 steps. If you set this to 10, the hill moves very fast!
        this.hillMoveInterval = config.hillMoveInterval || 100;
        
        // agentSpeed: How fast does the agent move each step?
        // Each time the agent takes an action, they move this many units.
        // Speed 0.5 means it takes 40 steps to cross the whole arena (20 / 0.5 = 40).
        this.agentSpeed = config.agentSpeed || 0.5;
        
        // ----- ADAPTIVE FEATURES -----
        // These make the environment CHANGE OVER TIME
        // This is what makes this "Adaptive" RL instead of normal RL
        
        // adaptiveInterval: Does the hill move at different speeds over time?
        // When true, the hill might move faster or slower as the game progresses.
        // This forces the agent to keep adapting instead of learning one pattern.
        this.adaptiveInterval = config.adaptiveInterval !== false;
        
        // adaptiveIntervalRange: How much can the move speed change?
        // min: fastest the hill can move (smallest number of steps between moves)
        // max: slowest the hill can move (largest number of steps between moves)
        // Example: { min: 40, max: 160 } means the hill moves every 40 to 160 steps
        this.adaptiveIntervalRange = config.adaptiveIntervalRange || { min: 40, max: 160 };
        
        // ----- INTERNAL STATE (Things that change during the game) -----
        // These are like the "current situation" in the playground
        
        // agentPos: Where is the agent right now?
        // x and z are coordinates on the ground. (0,0) is the center.
        this.agentPos = { x: 0, z: 0 };
        
        // goalPos: Where is the hill right now?
        // The agent wants to get close to this position.
        this.goalPos = { x: 0, z: 0 };
        
        // stepCount: How many total actions have happened in this episode?
        // Each time the agent moves (or stays), this increases by 1.
        this.stepCount = 0;
        
        // stepsSinceLastMove: How many steps since the hill last moved?
        // When this reaches currentInterval, the hill jumps to a new location.
        this.stepsSinceLastMove = 0;
        
        // currentInterval: How many steps until the NEXT hill move?
        // This can change over time (that's the adaptive part!)
        this.currentInterval = this.hillMoveInterval;
        
        // episodeReward: How many points has the agent earned in this episode?
        // Resets to 0 at the start of each new episode.
        this.episodeReward = 0;
        
        // totalReward: How many points has the agent earned across ALL episodes?
        // Never resets. Shows long-term performance.
        this.totalReward = 0;
        
        // ----- METRICS FOR TRACKING ADAPTATION -----
        // These help us measure HOW WELL the agent is adapting
        
        // goalChanges: A list of every time the hill moved
        // Each entry has: { step: which step it happened, position: where it moved to }
        this.goalChanges = [];
        
        // adaptationTimes: How fast did the agent find the hill after it moved?
        // Each time the hill moves, we record how many steps until the agent reached it.
        // Smaller numbers = faster adaptation = GOOD!
        this.adaptationTimes = [];
        
        // lastZoneEntryStep: When was the last time the agent was in the hill?
        // Used to calculate how long it takes to re-enter after the hill moves.
        this.lastZoneEntryStep = 0;
        
        // episodeCount: How many games (episodes) have been played?
        // Each episode is one complete game from start to finish.
        this.episodeCount = 0;
        
        // recentRewards: A list of recent rewards for performance tracking
        this.recentRewards = [];
        
        // Start the first game immediately
        this.reset();
    }
    
    // ------------------------------------------------------------------------
    // RESET - START A NEW GAME
    // ------------------------------------------------------------------------
    // This function starts a fresh episode.
    // It puts the agent and hill in random starting positions.
    // It also MAY change the rules (adaptive interval) for the next episode.
    //
    // WHEN IS THIS CALLED?
    // -------------------
    // - At the very beginning of training
    // - After each episode ends (after 1000 steps)
    // - When you want to restart the game
    //
    // WHAT DOES IT RETURN?
    // -------------------
    // It returns an "observation" - what the agent can see at the start.
    //
    
    reset() {
        
        // ----- STEP 1: Place the agent randomly -----
        // We subtract 6 from arenaSize so the agent doesn't start right at the edge.
        // Math.random() gives a number between 0 and 1.
        // (Math.random() - 0.5) gives a number between -0.5 and 0.5.
        // Multiply by (arenaSize - 6) to spread across the arena.
        // 
        // EXAMPLE: If arenaSize = 20, then arenaSize - 6 = 14.
        // So agent starts between -7 and +7 in both x and z directions.
        this.agentPos = {
            x: (Math.random() - 0.5) * (this.arenaSize - 6),
            z: (Math.random() - 0.5) * (this.arenaSize - 6)
        };
        
        // ----- STEP 2: Place the hill randomly -----
        // The _getRandomPosition() helper (defined below) puts the hill somewhere.
        // It avoids placing it too close to the edges.
        this.goalPos = this._getRandomPosition();
        
        // ----- STEP 3: Reset all counters to zero -----
        this.stepCount = 0;
        this.stepsSinceLastMove = 0;
        this.episodeReward = 0;
        
        // ----- STEP 4: ADAPT THE ENVIRONMENT (KEY FEATURE!) -----
        // This is what makes this "Adaptive" RL.
        // After the first episode (episodeCount > 0), we might change the rules.
        // The hill might move faster, slower, or at random intervals.
        // The agent must adapt to these changing rules!
        if (this.adaptiveInterval && this.episodeCount > 0) {
            this._adaptInterval();
        }
        
        // ----- STEP 5: Increase episode counter -----
        this.episodeCount++;
        
        // ----- STEP 6: Return what the agent can see -----
        return this._getObservation();
    }
    
    // ------------------------------------------------------------------------
    // STEP - LET THE AGENT MAKE ONE MOVE
    // ------------------------------------------------------------------------
    // This is the MOST IMPORTANT function in the environment.
    // It does FIVE things in order:
    // 1. Move the agent according to the action
    // 2. Calculate how far the agent is from the hill
    // 3. Calculate the reward (points earned)
    // 4. Check if the hill should move
    // 5. Return everything the agent needs to learn
    //
    // ACTIONS (what the agent can do):
    // --------------------------------
    // 0 = Move UP (increase Z coordinate)
    // 1 = Move DOWN (decrease Z coordinate)
    // 2 = Move LEFT (decrease X coordinate)
    // 3 = Move RIGHT (increase X coordinate)
    // 4 = STAY (don't move at all)
    //
    // QUESTION FOR YOU: Why would an agent ever choose to STAY?
    // ANSWER: Sometimes staying in the hill is better than moving away!
    //
    
    step(action) {
        
        // ----- STEP 1: Translate action number into movement -----
        // This is like a lookup table: action 0 means "move up"
        const moves = {
            0: { x: 0, z: 1 },   // Up: increase Z
            1: { x: 0, z: -1 },  // Down: decrease Z
            2: { x: -1, z: 0 },  // Left: decrease X
            3: { x: 1, z: 0 },   // Right: increase X
            4: { x: 0, z: 0 }    // Stay: don't change anything
        };
        
        const move = moves[action];
        
        // ----- STEP 2: Calculate new position -----
        // newX = oldX + (direction * speed)
        let newX = this.agentPos.x + move.x * this.agentSpeed;
        let newZ = this.agentPos.z + move.z * this.agentSpeed;
        
        // ----- STEP 3: Apply BOUNDARIES (can't leave the arena) -----
        // Math.max picks the larger number, Math.min picks the smaller.
        // This keeps the agent inside the arena.
        // Example: If newX = 19 and arenaSize = 20, Math.min(19, 18) = 18
        // The -2 creates a small wall so the agent can't touch the very edge.
        newX = Math.max(-this.arenaSize + 2, Math.min(this.arenaSize - 2, newX));
        newZ = Math.max(-this.arenaSize + 2, Math.min(this.arenaSize - 2, newZ));
        
        // Update the agent's actual position
        this.agentPos = { x: newX, z: newZ };
        
        // ----- STEP 4: Calculate distance to the hill -----
        // This uses the Pythagorean theorem: distance = sqrt((x1-x2)^2 + (z1-z2)^2)
        const distance = this._getDistance();
        
        // Is the agent close enough to be "in the zone"?
        const inZone = distance < this.hillRadius;
        
        // ----- STEP 5: Calculate REWARD (how many points?) -----
        // The _calculateReward function decides how many points the agent gets.
        // This is where we encourage good behavior and discourage bad behavior.
        let reward = this._calculateReward(distance, inZone);
        
        // ----- STEP 6: Check if the hill should move -----
        // The hill moves after a certain number of steps (currentInterval)
        this.stepsSinceLastMove++;
        let goalMoved = false;
        
        if (this.stepsSinceLastMove >= this.currentInterval) {
            // Move the hill to a random new position
            this.goalPos = this._getRandomPosition();
            this.stepsSinceLastMove = 0;
            goalMoved = true;
            
            // Record this goal change for metrics
            this.goalChanges.push({
                step: this.stepCount,
                position: { ...this.goalPos }
            });
        }
        
        // ----- STEP 7: Track adaptation speed -----
        // When the hill moves, we record how many steps it took to get back to it
        if (goalMoved) {
            const stepsSinceLastZone = this.stepCount - this.lastZoneEntryStep;
            this.adaptationTimes.push(stepsSinceLastZone);
        }
        
        // ----- STEP 8: Track when agent is in the zone -----
        if (inZone) {
            this.lastZoneEntryStep = this.stepCount;
        }
        
        // ----- STEP 9: Update counters -----
        this.stepCount++;
        this.episodeReward += reward;
        this.totalReward += reward;
        
        // ----- STEP 10: Check if episode is over -----
        // Episode ends after 1000 steps (prevents infinite games)
        const done = this.stepCount >= 1000;
        
        // ----- STEP 11: Return everything the agent needs -----
        return {
            observation: this._getObservation(),  // What the agent sees now
            reward: reward,                       // How many points earned
            done: done,                           // Is the game over?
            info: {                               // Extra information
                distance: distance,
                inZone: inZone,
                goalMoved: goalMoved,
                episodeReward: this.episodeReward,
                stepsRemaining: 1000 - this.stepCount,
                currentInterval: this.currentInterval
            }
        };
    }
    
    // ------------------------------------------------------------------------
    // CALCULATE REWARD - HOW MANY POINTS DOES THE AGENT GET?
    // ------------------------------------------------------------------------
    // This is where we tell the agent what is GOOD and what is BAD.
    // 
    // THE REWARD RULES:
    // -----------------
    // +1.0 point for each step INSIDE the hill (GOOD - encourages staying)
    // -0.01 point for each step (small penalty - encourages finishing faster)
    // + Speed bonus for reaching the hill quickly after it moves (ADAPTIVE!)
    // + Staying bonus for remaining in the hill (ADAPTIVE!)
    //
    // WHY IS THIS ADAPTIVE?
    // --------------------
    // Notice how the reward changes based on "stepsSinceLastMove":
    // - Early after hill moves (steps < 20): We reward SPEED (distance bonus)
    // - Later (steps >= 20): We reward STAYING (extra points for being in zone)
    // 
    // This teaches the agent: "When the hill just moved, GET THERE FAST.
    // Once you're there, STAY THERE."
    //
    
    _calculateReward(distance, inZone) {
        let reward = 0;
        
        // ----- RULE 1: Reward for being IN the hill -----
        // This is the main way the agent earns points.
        // +1 point every step while inside the golden zone.
        if (inZone) {
            reward += 1.0;
        }
        
        // ----- RULE 2: Small time penalty -----
        // This encourages the agent to finish quickly.
        // -0.01 points every step (even if in zone).
        // Over 1000 steps, this adds up to -10 points.
        reward -= 0.01;
        
        // ----- RULE 3: ADAPTIVE REWARD SHAPING -----
        // This changes based on how long ago the hill moved!
        
        if (this.stepsSinceLastMove < 20) {
            // EARLY PHASE: The hill just moved recently (within last 20 steps)
            // We want the agent to REACH THE NEW HILL QUICKLY.
            // So we give a BONUS based on how CLOSE they are to the hill.
            // The closer they are, the bigger the bonus.
            // 
            // Formula: (1 - distance/arenaSize) * 0.5
            // Example: If distance = 0 (on the hill), bonus = (1 - 0) * 0.5 = 0.5
            // Example: If distance = 10 (far away), bonus = (1 - 0.5) * 0.5 = 0.25
            const speedBonus = Math.max(0, (1 - distance / this.arenaSize) * 0.5);
            reward += speedBonus;
        } else {
            // LATE PHASE: The hill has been in the same place for a while
            // We want the agent to STAY IN THE HILL.
            // So we give extra points for each step they remain inside.
            if (inZone) {
                reward += 0.1;  // Staying bonus
            }
        }
        
        return reward;
    }
    
    // ------------------------------------------------------------------------
    // ADAPT INTERVAL - CHANGE HOW FAST THE HILL MOVES
    // ------------------------------------------------------------------------
    // THIS IS THE KEY ADAPTIVE FEATURE!
    // 
    // In normal Reinforcement Learning, the rules never change.
    // The agent learns ONE strategy and uses it forever.
    //
    // In ADAPTIVE Reinforcement Learning, the rules CHANGE over time.
    // The agent must KEEP LEARNING and ADAPTING.
    //
    // This function changes the hillMoveInterval (how many steps between hill moves).
    // It can make the hill move FASTER (harder), SLOWER (easier), or RANDOMLY.
    //
    // THE FOUR STRATEGIES:
    // --------------------
    // 1. 'random'  - Completely random interval between min and max
    // 2. 'faster'  - Make the hill move MORE OFTEN (harder for agent)
    // 3. 'slower'  - Make the hill move LESS OFTEN (easier for agent)
    // 4. 'cyclic'  - Oscillates like a wave (fast, slow, fast, slow)
    //
    // QUESTION FOR YOU: Which strategy is hardest for the agent?
    // ANSWER: 'random' because there's no pattern to learn!
    //
    
    _adaptInterval() {
        const oldInterval = this.currentInterval;
        
        // Choose one of four adaptation strategies randomly
        const strategies = ['random', 'faster', 'slower', 'cyclic'];
        const strategy = strategies[Math.floor(Math.random() * strategies.length)];
        
        switch(strategy) {
            case 'random':
                // Completely random: Any number between min and max
                this.currentInterval = Math.floor(
                    this.adaptiveIntervalRange.min + 
                    Math.random() * (this.adaptiveIntervalRange.max - this.adaptiveIntervalRange.min)
                );
                break;
                
            case 'faster':
                // Make hill move MORE frequently (multiply by 0.7 = 30% faster)
                // Example: 100 steps * 0.7 = 70 steps between moves
                this.currentInterval = Math.max(
                    this.adaptiveIntervalRange.min,
                    this.currentInterval * 0.7
                );
                break;
                
            case 'slower':
                // Make hill move LESS frequently (multiply by 1.5 = 50% slower)
                // Example: 100 steps * 1.5 = 150 steps between moves
                this.currentInterval = Math.min(
                    this.adaptiveIntervalRange.max,
                    this.currentInterval * 1.5
                );
                break;
                
            case 'cyclic':
                // Creates a wave pattern using sine function
                // episodeCount * 0.1 creates a slow oscillation
                // sin goes from -1 to +1, we map to [min, max]
                const cycle = Math.sin(this.episodeCount * 0.1);
                const range = this.adaptiveIntervalRange.max - this.adaptiveIntervalRange.min;
                // ((cycle + 1) / 2) converts -1..+1 to 0..1
                this.currentInterval = Math.floor(
                    this.adaptiveIntervalRange.min + ((cycle + 1) / 2) * range
                );
                break;
        }
        
        // Log the change so we can see adaptation happening
        console.log(`ENVIRONMENT ADAPTED: Goal move interval ${oldInterval} -> ${this.currentInterval} steps`);
    }
    
    // ------------------------------------------------------------------------
    // HELPER FUNCTIONS
    // ------------------------------------------------------------------------
    
    // Get a random position within the arena
    // The margin (3 units) keeps the hill away from the edges
    _getRandomPosition() {
        const margin = 3;
        return {
            x: (Math.random() - 0.5) * (this.arenaSize - margin * 2),
            z: (Math.random() - 0.5) * (this.arenaSize - margin * 2)
        };
    }
    
    // Calculate distance from agent to hill using Pythagorean theorem
    // distance = square root of (dx squared + dz squared)
    _getDistance() {
        const dx = this.agentPos.x - this.goalPos.x;
        const dz = this.agentPos.z - this.goalPos.z;
        return Math.sqrt(dx * dx + dz * dz);
    }
    
    // ------------------------------------------------------------------------
    // GET OBSERVATION - WHAT THE AGENT CAN "SEE"
    // ------------------------------------------------------------------------
    // The agent cannot see everything in the world.
    // It can only see SIX numbers that describe its situation:
    //
    // 1. agent_x : Agent's X position (normalized from -1 to 1)
    // 2. agent_z : Agent's Z position (normalized from -1 to 1)
    // 3. goal_x  : Hill's X position (normalized from -1 to 1)
    // 4. goal_z  : Hill's Z position (normalized from -1 to 1)
    // 5. time_norm: How long until hill moves (0 = just moved, 1 = about to move)
    // 6. in_zone_flag: 1 if in hill, 0 if not (binary)
    //
    // NORMALIZATION EXPLAINED:
    // -----------------------
    // Normalization means converting numbers to a standard range (-1 to 1).
    // Original positions range from -18 to +18.
    // Dividing by arenaSize (20) gives roughly -0.9 to +0.9.
    // This helps the agent learn because all inputs are on the same scale.
    //
    
    _getObservation() {
        // Normalize positions: divide by arena size to get range roughly -1 to 1
        const normX = this.agentPos.x / this.arenaSize;
        const normZ = this.agentPos.z / this.arenaSize;
        const goalNormX = this.goalPos.x / this.arenaSize;
        const goalNormZ = this.goalPos.z / this.arenaSize;
        
        // Time until next hill move (0 = just moved, 1 = about to move)
        const timeNorm = this.stepsSinceLastMove / this.currentInterval;
        
        // Is the agent in the hill? (1 = yes, 0 = no)
        const inZone = this._getDistance() < this.hillRadius ? 1 : 0;
        
        // Return as Float32Array (efficient for machine learning)
        return new Float32Array([normX, normZ, goalNormX, goalNormZ, timeNorm, inZone]);
    }
    
    // ------------------------------------------------------------------------
    // GET METRICS - HOW WELL IS THE AGENT ADAPTING?
    // ------------------------------------------------------------------------
    // Returns performance data that can be used for visualization or logging
    //
    // averageAdaptationTime: How many steps it takes the agent to find the hill after it moves
    //   - Small number = GOOD (agent adapts quickly)
    //   - Large number = BAD (agent takes long to find new hill)
    //
    
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


// ============================================================================
// PART 2: TENSORFLOW.JS DEEP Q-NETWORK AGENT
// ============================================================================
//
// WHAT IS A DEEP Q-NETWORK (DQN)?
// -------------------------------
// DQN is a type of Reinforcement Learning that uses a NEURAL NETWORK
// to approximate Q-values instead of storing them in a table.
//
// WHY USE A NEURAL NETWORK?
// -------------------------
// - Handles CONTINUOUS states directly (no need to discretize)
// - Can GENERALIZE - if it learns to go left, it knows going right is different
// - Much more powerful for complex problems
// - Can handle millions of possible states (where a table would be too big)
//
// THE NEURAL NETWORK ARCHITECTURE:
// -------------------------------
// Input (6) -> Dense(64, ReLU) -> Dense(64, ReLU) -> Output(5)
//
// WHAT IS ReLU?
// -------------
// ReLU = Rectified Linear Unit. It's a simple function: f(x) = max(0, x)
// If input is negative, output 0. If input is positive, output same value.
// This helps the network learn non-linear patterns (curved relationships).
//
// THE TWO NETWORKS (Main and Target):
// ----------------------------------
// - Main network: Learns and updates every step
// - Target network: Used to calculate targets, updates slowly
// 
// WHY TWO NETWORKS?
// -----------------
// If you use the same network to both predict and calculate targets,
// you create a feedback loop. The network chases its own predictions.
// Using a separate target network stabilizes training.
//
// EXPERIENCE REPLAY:
// -----------------
// Instead of learning from each experience once and forgetting it,
// we store experiences in a "memory" and learn from random batches.
// This prevents catastrophic forgetting (learning new things and forgetting old ones).
//

class TensorFlowDQNAgent {
    
    // ------------------------------------------------------------------------
    // CONSTRUCTOR - BUILD THE NEURAL NETWORK
    // ------------------------------------------------------------------------
    // Creates the neural network, optimizer, and memory buffer.
    //
    // LEARNING PARAMETERS:
    // --------------------
    // learningRate (alpha): How much to trust new experiences (0.001)
    //   Neural networks need smaller learning rates than Q-tables (0.001 vs 0.1)
    //   because big changes can destabilize the network.
    //
    // discountFactor (gamma): How much to value future rewards (0.95)
    //   - 0.95 means "future rewards are almost as good as current rewards"
    //   - Encourages long-term planning
    //
    // epsilon: How often to take random actions (0.3)
    //   - 0.3 = take random action 30% of the time
    //   - High epsilon = more EXPLORATION (trying new things)
    //   - Low epsilon = more EXPLOITATION (using what you know)
    //
    // memorySize: How many experiences to remember (10000)
    // batchSize: How many experiences to learn from at once (32)
    // targetUpdateFreq: How often to update target network (100 steps)
    //
    
    constructor(config = {}) {
        
        // ----- LEARNING PARAMETERS -----
        this.learningRate = config.learningRate || 0.001;   // Neural networks need smaller learning rates
        this.discountFactor = config.discountFactor || 0.95;
        this.epsilon = config.epsilon || 0.3;
        this.epsilonMin = config.epsilonMin || 0.05;
        this.epsilonDecay = config.epsilonDecay || 0.995;
        
        // ----- EXPERIENCE REPLAY MEMORY -----
        this.memory = [];
        this.memorySize = config.memorySize || 10000;
        this.batchSize = config.batchSize || 32;
        
        // ----- TARGET NETWORK UPDATE FREQUENCY -----
        this.targetUpdateFreq = config.targetUpdateFreq || 100;
        this.trainStep = 0;
        
        // ----- TRACKING METRICS -----
        this.recentRewards = [];
        this.episodeRewards = [];
        this.totalSteps = 0;
        this.lastLoss = 0;
        
        // ----- BUILD THE NEURAL NETWORKS -----
        // We need TWO networks: main network (learns) and target network (for stability)
        this.mainNetwork = this._buildNetwork();
        this.targetNetwork = this._buildNetwork();
        
        // Copy weights from main to target (so they start identical)
        this._updateTargetNetwork();
        
        // ----- CREATE THE OPTIMIZER -----
        // Adam is a popular optimizer that adapts the learning rate automatically
        this.optimizer = tf.train.adam(this.learningRate);
        
        console.log('TensorFlow DQN Agent Created');
        console.log(`  - Neural Network: 6 inputs -> 64 -> 64 -> 5 outputs`);
        console.log(`  - Memory Size: ${this.memorySize}`);
        console.log(`  - Batch Size: ${this.batchSize}`);
    }
    
    // ------------------------------------------------------------------------
    // BUILD THE NEURAL NETWORK
    // ------------------------------------------------------------------------
    // This creates a sequential model (layers stacked on top of each other)
    //
    // LAYER 1: Dense(64, ReLU, inputShape=[6])
    //   - Takes 6 inputs (agent x,z, goal x,z, time, inZone)
    //   - Has 64 neurons
    //   - Uses ReLU activation (f(x) = max(0, x))
    //
    // LAYER 2: Dense(64, ReLU)
    //   - Takes 64 inputs from previous layer
    //   - Has 64 neurons
    //   - Uses ReLU activation
    //
    // LAYER 3: Dense(5, linear)
    //   - Takes 64 inputs from previous layer
    //   - Has 5 neurons (one for each action)
    //   - Uses linear activation (outputs raw numbers - Q-values)
    //
    // WHAT IS A DENSE LAYER?
    // ----------------------
    // A dense layer connects EVERY neuron in this layer to EVERY neuron in the next layer.
    // This is also called a "fully connected" layer.
    //
    // WHY 64 NEURONS?
    // ---------------
    // 64 is a common choice. More neurons = more learning capacity but slower.
    // 64 is a good balance for this problem.
    //
    
    _buildNetwork() {
        const model = tf.sequential();
        
        // Hidden layer 1: 64 neurons with ReLU activation
        model.add(tf.layers.dense({
            units: 64,
            activation: 'relu',
            inputShape: [6]
        }));
        
        // Hidden layer 2: 64 neurons with ReLU activation
        model.add(tf.layers.dense({
            units: 64,
            activation: 'relu'
        }));
        
        // Output layer: 5 neurons (one per action) with linear activation
        // Linear activation means we don't transform the outputs (raw Q-values)
        model.add(tf.layers.dense({
            units: 5,
            activation: 'linear'
        }));
        
        return model;
    }
    
    // ------------------------------------------------------------------------
    // COPY WEIGHTS FROM MAIN NETWORK TO TARGET NETWORK
    // ------------------------------------------------------------------------
    // The target network is a "slower" copy of the main network.
    // It helps stabilize training by providing consistent targets.
    //
    // HOW OFTEN DO WE UPDATE THE TARGET NETWORK?
    // -----------------------------------------
    // Every 100 training steps (targetUpdateFreq).
    // This means the target network lags behind the main network.
    // This lag is intentional and helps with stability.
    //
    
    _updateTargetNetwork() {
        const weights = this.mainNetwork.getWeights();
        this.targetNetwork.setWeights(weights);
    }
    
    // ------------------------------------------------------------------------
    // GET Q-VALUES FOR AN OBSERVATION
    // ------------------------------------------------------------------------
    // Runs the observation through the neural network to get Q-values for all actions
    //
    // INPUT: observation (array of 6 numbers)
    // OUTPUT: array of 5 numbers (Q-values for actions 0-4)
    //
    // EXAMPLE OUTPUT:
    // [10.5, 2.3, -1.2, 8.7, 5.0] means:
    //   - Moving Up is worth 10.5 points (BEST)
    //   - Moving Down is worth 2.3 points
    //   - Moving Left is worth -1.2 points (BAD)
    //   - Moving Right is worth 8.7 points
    //   - Staying is worth 5.0 points
    //
    
    _getQValues(observation) {
        // Convert observation to tensor (TensorFlow's data format)
        // tensor2d creates a 2D tensor: [1, 6] means 1 row, 6 columns
        const inputTensor = tf.tensor2d([observation], [1, 6]);
        
        // Run through the network (forward pass)
        const qValues = this.mainNetwork.predict(inputTensor);
        
        // Get the data as a regular JavaScript array
        const values = qValues.dataSync();
        
        // Clean up tensors to prevent memory leaks (very important!)
        inputTensor.dispose();
        qValues.dispose();
        
        return values;
    }
    
    // ------------------------------------------------------------------------
    // SELECT ACTION USING EPSILON-GREEDY POLICY
    // ------------------------------------------------------------------------
    // With probability epsilon: take random action (explore)
    // With probability 1-epsilon: take best action according to network (exploit)
    //
    // ADAPTIVE FEATURE: When goal moves, temporarily increase exploration
    // This helps the agent find the new hill location faster.
    //
    
    act(observation, info = {}) {
        
        // ----- ADAPTIVE BEHAVIOR: Explore more when hill moves -----
        // When the goal moves, old strategies might not work anymore.
        // We temporarily increase epsilon by 0.2 (but never above 0.5)
        if (info.goalMoved) {
            const boostedEpsilon = Math.min(0.5, this.epsilon + 0.2);
            if (Math.random() < boostedEpsilon) {
                return Math.floor(Math.random() * 5);  // Random action
            }
        }
        
        // ----- NORMAL EPSILON-GREEDY -----
        if (Math.random() < this.epsilon) {
            // EXPLORE: Take a random action
            return Math.floor(Math.random() * 5);
        } else {
            // EXPLOIT: Take the best action according to the neural network
            const qValues = this._getQValues(observation);
            return qValues.indexOf(Math.max(...qValues));
        }
    }
    
    // ------------------------------------------------------------------------
    // TRAIN THE NETWORK ON A SINGLE TRANSITION
    // ------------------------------------------------------------------------
    // This function does two things:
    // 1. Stores the experience in memory (for experience replay)
    // 2. Periodically trains on batches of past experiences
    //
    // The actual learning happens in _replay(), which is called when we have
    // enough experiences in memory.
    //
    
    train(observation, action, reward, nextObservation, done, info = {}) {
        
        // ----- STEP 1: Store experience in memory -----
        this.memory.push({
            observation: observation.slice(),  // Copy the array (don't store reference)
            action: action,
            reward: reward,
            nextObservation: nextObservation.slice(),
            done: done
        });
        
        // ----- STEP 2: Limit memory size (remove oldest if too big) -----
        if (this.memory.length > this.memorySize) {
            this.memory.shift();  // Remove the oldest experience
        }
        
        // ----- STEP 3: Track rewards for performance monitoring -----
        this.recentRewards.push(reward);
        if (this.recentRewards.length > 100) {
            this.recentRewards.shift();  // Keep only last 100 rewards
        }
        
        // ----- STEP 4: Train on a batch of experiences -----
        // We wait until we have at least batchSize experiences before training
        if (this.memory.length >= this.batchSize) {
            this._replay();
        }
        
        // ----- STEP 5: Update target network periodically -----
        this.trainStep++;
        if (this.trainStep % this.targetUpdateFreq === 0) {
            this._updateTargetNetwork();
        }
        
        // ----- STEP 6: Adaptive epsilon decay -----
        this._adaptiveEpsilonDecay();
        
        this.totalSteps++;
    }
    
    // ------------------------------------------------------------------------
    // EXPERIENCE REPLAY - TRAIN ON BATCH OF PAST EXPERIENCES
    // ------------------------------------------------------------------------
    // This is the heart of DQN learning.
    //
    // HOW IT WORKS:
    // -------------
    // 1. Sample a random batch of experiences from memory
    // 2. For each experience, calculate the target Q-value using the Bellman equation
    // 3. Train the network to predict these target values
    //
    // WHY RANDOM BATCHES?
    // -------------------
    // If we train on consecutive experiences, the network might overfit to recent events.
    // Random sampling breaks correlations and leads to better learning.
    //
    // WHY USE THE TARGET NETWORK FOR TARGETS?
    // ---------------------------------------
    // The target network provides stable targets. If we used the main network,
    // the targets would change every step, making training unstable.
    //
    
    async _replay() {
        
        // ----- STEP 1: Sample random batch from memory -----
        const batch = [];
        const indices = new Set();
        
        while (indices.size < this.batchSize) {
            indices.add(Math.floor(Math.random() * this.memory.length));
        }
        
        for (const idx of indices) {
            batch.push(this.memory[idx]);
        }
        
        // ----- STEP 2: Prepare current states and next states -----
        const currentStates = batch.map(e => e.observation);
        const nextStates = batch.map(e => e.nextObservation);
        
        // ----- STEP 3: Get current Q-values from MAIN network -----
        const currentQTensor = this.mainNetwork.predict(
            tf.tensor2d(currentStates, [this.batchSize, 6])
        );
        const currentQValues = await currentQTensor.array();
        
        // ----- STEP 4: Get next Q-values from TARGET network -----
        // Notice: We use the TARGET network here, not the main network!
        const nextQTensor = this.targetNetwork.predict(
            tf.tensor2d(nextStates, [this.batchSize, 6])
        );
        const nextQValues = await nextQTensor.array();
        
        // ----- STEP 5: Calculate target Q-values using Bellman equation -----
        // Target = reward + gamma * max(next Q-value) if not done
        // Target = reward if done
        const targetQValues = [];
        
        for (let i = 0; i < this.batchSize; i++) {
            // Start with the current Q-values
            const targetQ = [...currentQValues[i]];
            const action = batch[i].action;
            const reward = batch[i].reward;
            const done = batch[i].done;
            
            if (done) {
                // Episode ended: target is just the reward received
                targetQ[action] = reward;
            } else {
                // Bellman equation: target = reward + gamma * max(next Q)
                const maxNextQ = Math.max(...nextQValues[i]);
                targetQ[action] = reward + this.discountFactor * maxNextQ;
            }
            
            targetQValues.push(targetQ);
        }
        
        // ----- STEP 6: Train the network to predict these targets -----
        const xs = tf.tensor2d(currentStates, [this.batchSize, 6]);
        const ys = tf.tensor2d(targetQValues, [this.batchSize, 5]);
        
        const loss = await this._trainBatch(xs, ys);
        this.lastLoss = loss;
        
        // ----- STEP 7: Clean up tensors (prevent memory leaks) -----
        currentQTensor.dispose();
        nextQTensor.dispose();
        xs.dispose();
        ys.dispose();
    }
    
    // ------------------------------------------------------------------------
    // TRAIN ON A BATCH - THE ACTUAL WEIGHT UPDATE
    // ------------------------------------------------------------------------
    // This does the actual gradient descent step.
    //
    // WHAT IS GRADIENT DESCENT?
    // -------------------------
    // Gradient descent is how neural networks learn.
    // 
    // 1. Make a prediction (forward pass)
    // 2. Calculate how wrong we were (loss)
    // 3. Calculate the gradient (direction to improve)
    // 4. Update weights in that direction
    //
    // The optimizer.minimize function does steps 3 and 4 automatically.
    //
    
    async _trainBatch(states, targets) {
        return new Promise((resolve) => {
            this.optimizer.minimize(() => {
                // Forward pass: get predictions
                const predictions = this.mainNetwork.apply(states, { training: true });
                
                // Calculate loss: mean squared error between predictions and targets
                const loss = tf.losses.meanSquaredError(targets, predictions);
                
                // Return the loss value for logging
                resolve(loss.dataSync()[0]);
                
                return loss;
            });
        });
    }
    
    // ------------------------------------------------------------------------
    // ADAPTIVE EPSILON DECAY
    // ------------------------------------------------------------------------
    // In normal Q-learning, epsilon decays at a fixed rate regardless of performance.
    // In ADAPTIVE Q-learning, we adjust epsilon based on how well the agent is doing.
    //
    // THE RULES:
    // ----------
    // - If average reward > 0.3: Doing well! Decay epsilon FASTER (exploit more)
    // - If average reward < 0.1: Doing poorly! Decay epsilon SLOWER (explore more)
    // - Otherwise: Decay at normal rate
    //
    // WHY THIS WORKS:
    // --------------
    // When the agent is winning, it should trust its knowledge and exploit.
    // When the agent is losing, it needs to try new strategies and explore.
    //
    
    _adaptiveEpsilonDecay() {
        if (this.recentRewards.length < 50) {
            // Not enough data yet, use normal decay
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
                // Average performance: normal decay
                this.epsilon *= this.epsilonDecay;
            }
        }
        
        // Never go below minimum exploration rate
        this.epsilon = Math.max(this.epsilonMin, this.epsilon);
    }
    
    // ------------------------------------------------------------------------
    // END EPISODE - RECORD RESULTS
    // ------------------------------------------------------------------------
    // Called at the end of each episode.
    // Records the total reward and may boost exploration if performance is poor.
    //
    
    endEpisode(totalReward) {
        this.episodeRewards.push(totalReward);
        
        // Keep only last 100 episodes (prevents memory bloat)
        if (this.episodeRewards.length > 100) {
            this.episodeRewards.shift();
        }
        
        // If recent performance is very poor, boost exploration
        const avgRecent = this.episodeRewards.slice(-5).reduce((a,b) => a + b, 0) / 5;
        if (avgRecent < 10 && this.epsilon < 0.2) {
            this.epsilon = 0.3;
            console.log(`Exploration boosted due to poor performance`);
        }
    }
    
    // ------------------------------------------------------------------------
    // GET METRICS - REPORT AGENT'S PERFORMANCE
    // ------------------------------------------------------------------------
    // Returns performance data for visualization or logging
    //
    
    getMetrics() {
        const avgRecentReward = this.recentRewards.length > 0
            ? this.recentRewards.reduce((a,b) => a + b, 0) / this.recentRewards.length
            : 0;
        
        return {
            epsilon: this.epsilon.toFixed(3),
            memorySize: this.memory.length,
            totalSteps: this.totalSteps,
            averageReward: avgRecentReward.toFixed(2),
            lastLoss: this.lastLoss.toFixed(4)
        };
    }
}


// ============================================================================
// PART 3: THE TRAINER CLASS
// ============================================================================
//
// WHAT IS A TRAINER?
// ------------------
// The trainer orchestrates the whole learning process.
// It runs episodes, calls the environment and agent at the right times,
// and broadcasts updates for visualization.
//
// THE TRAINING LOOP:
// -----------------
// For each episode:
//   1. Reset the environment (start new game)
//   2. For each step (up to 1000):
//      a. Agent chooses an action
//      b. Environment processes the action
//      c. Agent learns from the result
//      d. Broadcast update for visualization
//   3. Record episode results
//   4. Broadcast episode complete
//
// WHY DO WE NEED A TRAINER?
// -------------------------
// The trainer keeps the environment and agent separate.
// This is good software design because:
// - You can swap different agents (RandomAgent, DQNAgent, etc.)
// - You can swap different environments
// - You can add logging, visualization, or metrics without changing the core logic
//

class AdaptiveRLTrainer {
    
    constructor(env, agent) {
        this.env = env;           // The game environment
        this.agent = agent;       // The learning agent
        this.trainingActive = false;
        this.currentEpisode = 0;
        this.listeners = [];      // Functions to call on updates
        this.rewardHistory = [];
    }
    
    // ------------------------------------------------------------------------
    // EVENT LISTENERS - FOR VISUALIZATION
    // ------------------------------------------------------------------------
    // Other code (like Three.js visualization) can register to receive updates
    // every time something happens in training.
    //
    // EXAMPLE:
    // trainer.onUpdate((data) => {
    //     if (data.type === 'step') {
    //         updateVisualization(data.observation);
    //     }
    // });
    //
    
    onUpdate(callback) {
        this.listeners.push(callback);
    }
    
    _notify(data) {
        this.listeners.forEach(cb => cb(data));
    }
    
    // ------------------------------------------------------------------------
    // TRAIN - RUN MULTIPLE EPISODES
    // ------------------------------------------------------------------------
    // This is the main training loop. It runs for the specified number of episodes.
    // The 'async' keyword allows the function to pause (await) without freezing the browser.
    //
    
    async train(numEpisodes, stepsPerEpisode = 500) {
        this.trainingActive = true;
        
        for (let ep = 0; ep < numEpisodes && this.trainingActive; ep++) {
            const episodeResult = await this._runEpisode(stepsPerEpisode);
            this.rewardHistory.push(episodeResult);
            
            // Notify listeners that an episode completed
            this._notify({
                type: 'episode_complete',
                episode: this.currentEpisode,
                reward: episodeResult.totalReward,
                metrics: this.env.getMetrics(),
                agentMetrics: this.agent.getMetrics(),
                adaptationTime: episodeResult.adaptationTime
            });
            
            // Small delay prevents UI freezing (gives browser time to render)
            await this._sleep(10);
        }
        
        this.trainingActive = false;
        return this.rewardHistory;
    }
    
    // ------------------------------------------------------------------------
    // RUN EPISODE - PLAY ONE COMPLETE GAME
    // ------------------------------------------------------------------------
    // Runs a single episode from reset to done.
    // Returns statistics about the episode.
    //
    
    async _runEpisode(maxSteps) {
        let observation = this.env.reset();  // Start new game
        let done = false;
        let step = 0;
        let totalReward = 0;
        let adaptationTimes = [];
        
        while (!done && step < maxSteps) {
            // Agent decides what to do
            const action = this.agent.act(observation, { goalMoved: false });
            
            // Environment processes the action
            const result = this.env.step(action);
            
            // Agent learns from what happened
            this.agent.train(
                observation, 
                action, 
                result.reward, 
                result.observation, 
                result.done,
                { goalMoved: result.info.goalMoved }
            );
            
            totalReward += result.reward;
            
            // Track how long it takes to adapt after hill moves
            if (result.info.goalMoved) {
                adaptationTimes.push(result.info.stepsRemaining);
            }
            
            // Broadcast this step for visualization
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
    
    // ------------------------------------------------------------------------
    // STOP - HALT TRAINING
    // ------------------------------------------------------------------------
    
    stop() {
        this.trainingActive = false;
    }
    
    // ------------------------------------------------------------------------
    // SLEEP - PAUSE FOR A MOMENT
    // ------------------------------------------------------------------------
    // Returns a Promise that resolves after 'ms' milliseconds.
    // This is how we prevent the browser from freezing during long training runs.
    //
    
    _sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
    
    // ------------------------------------------------------------------------
    // GET SUMMARY - FINAL STATISTICS
    // ------------------------------------------------------------------------
    
    getSummary() {
        const avgReward = this.rewardHistory.reduce((a,b) => a + b.totalReward, 0) / this.rewardHistory.length;
        const bestReward = Math.max(...this.rewardHistory.map(r => r.totalReward));
        
        return {
            episodesCompleted: this.currentEpisode,
            averageReward: avgReward,
            bestReward: bestReward,
            finalEpsilon: this.agent.epsilon
        };
    }
}


// ============================================================================
// EXPORT FOR USE IN HTML
// ============================================================================
//
// This makes the three classes available globally.
// In your HTML file, you can now write:
//
//   let env = new TensorFlowRL.Environment()
//   let agent = new TensorFlowRL.DQNAgent()
//   let trainer = new TensorFlowRL.Trainer(env, agent)
//
// The 'window' object is the global object in browsers.
// Adding properties to it makes them available everywhere.
//

window.TensorFlowRL = {
    Environment: AdaptiveHillEnvironment,
    DQNAgent: TensorFlowDQNAgent,
    Trainer: AdaptiveRLTrainer
};

console.log('========================================');
console.log('TensorFlow.js Adaptive RL Module Loaded');
console.log('========================================');
console.log('Available classes:');
console.log('  - TensorFlowRL.Environment  (the game world)');
console.log('  - TensorFlowRL.DQNAgent     (neural network AI)');
console.log('  - TensorFlowRL.Trainer      (runs training)');
console.log('');
console.log('Quick start:');
console.log('  let env = new TensorFlowRL.Environment();');
console.log('  let agent = new TensorFlowRL.DQNAgent();');
console.log('  let trainer = new TensorFlowRL.Trainer(env, agent);');
console.log('  trainer.train(100);');
console.log('========================================');

})();  // End of self-executing function
