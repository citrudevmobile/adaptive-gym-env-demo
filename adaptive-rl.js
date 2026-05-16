/**
 * ADAPTIVE REINFORCEMENT LEARNING MODULE
 * ======================================
 * 
 * WHAT IS THIS?
 * ------------
 * This code teaches an AI agent how to play a game where the rules keep changing.
 * The game is "King of the Hill" - the agent must stand on a golden hill to earn points.
 * But the hill moves to random places every few seconds!
 * 
 * WHY IS THIS SPECIAL?
 * -------------------
 * Normal AI learns one fixed strategy. This AI must ADAPT when things change.
 * When the hill moves, the agent must notice, adjust its plan, and find the new hill.
 * 
 * HOW TO USE THIS CODE?
 * --------------------
 * 1. Include this file in your HTML: <script src="adaptive-rl.js"></script>
 * 2. Create an environment: let env = new AdaptiveRL.Environment()
 * 3. Create an agent: let agent = new AdaptiveRL.Agent()
 * 4. Create a trainer: let trainer = new AdaptiveRL.Trainer(env, agent)
 * 5. Start training: trainer.train(100)  // trains for 100 episodes
 * 
 * WHAT YOU WILL LEARN FROM READING THIS CODE:
 * -------------------------------------------
 * - What is an "environment" in RL (the game world)
 * - What is an "agent" (the learner/player)
 * - What are "states" (what the agent sees)
 * - What are "actions" (what the agent can do)
 * - What are "rewards" (feedback for good/bad actions)
 * - What is "Q-learning" (a method to learn from rewards)
 * - What is "exploration vs exploitation" (trying new things vs using what you know)
 * - How to make an agent ADAPT when the environment changes
 * 
 * @author Evans Enonchong
 * @version 2.0 - Educational Edition with Complete Documentation
 */

// ============================================================================
// PART 1: THE ENVIRONMENT CLASS
// ============================================================================
// 
// WHAT IS AN ENVIRONMENT?
// -----------------------
// The environment is the GAME WORLD. It contains all the rules:
// - Where is the agent?
// - Where is the hill (goal)?
// - What happens when the agent moves?
// - How many points does the agent get?
// - When does the hill move?
//
// THINK OF IT LIKE THIS:
// ---------------------
// If you were playing hide and seek, the environment is:
// - The park you play in (boundaries)
// - Where the "base" is (the hill)
// - Where you are standing
// - The rules about tagging
// - The points you earn for reaching base
//
// THE ENVIRONMENT HAS THREE MAIN JOBS:
// ------------------------------------
// 1. reset() - Start a new game
// 2. step(action) - Let the agent take one move, then tell them what happened
// 3. _getObservation() - Tell the agent what they can "see"
//

class AdaptiveHillEnvironment {
    
    // ------------------------------------------------------------------------
    // THE CONSTRUCTOR - SETTING UP THE GAME WORLD
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
        console.log(`ENVIRONMENT ADAPTED: Goal move interval ${oldInterval} → ${this.currentInterval} steps`);
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
// PART 2: THE ADAPTIVE DQN AGENT
// ============================================================================
//
// WHAT IS AN AGENT?
// -----------------
// The agent is the LEARNER. It makes decisions, gets rewards, and tries to get better.
// Think of it as the "brain" that controls the blue character.
//
// HOW DOES THE AGENT LEARN?
// -------------------------
// The agent uses a method called Q-LEARNING.
// Q-learning works like this:
//   1. Agent looks at the current situation (observation)
//   2. Agent guesses how good each action would be (Q-values)
//   3. Agent takes the best action (or sometimes a random action to explore)
//   4. Agent gets a reward from the environment
//   5. Agent updates its guesses based on what actually happened
//
// THE Q-LEARNING FORMULA (Bellman Equation):
// ------------------------------------------
// New Q-value = Old Q-value + Learning Rate * (Reward + Discount * Best Future Q - Old Q-value)
//
// This formula has special meaning:
// - "Reward" = points received right now
// - "Discount * Best Future Q" = estimated future points (with discount for uncertainty)
// - The difference is the "surprise" or "error"
// - Learning Rate controls how much we trust new information
//
// WHAT MAKES THIS AGENT "ADAPTIVE"?
// ---------------------------------
// 1. When the hill moves, the agent EXPLORES MORE (temporarily increases randomness)
// 2. The exploration rate (epsilon) changes based on PERFORMANCE (explore more when losing)
// 3. The learning rate changes based on PERFORMANCE TRENDS (learn faster when confused)
//

class AdaptiveDQNAgent {
    
    // ------------------------------------------------------------------------
    // CONSTRUCTOR - SET UP THE AGENT'S BRAIN
    // ------------------------------------------------------------------------
    //
    // LEARNING PARAMETERS (How the agent learns):
    // -------------------------------------------
    // learningRate (alpha): How much to trust new experiences (0 to 1)
    //   - 0.1 means "change Q-values by 10% toward the new information"
    //   - Higher = learns faster but might forget old knowledge
    //   - Lower = learns slower but more stable
    //
    // discountFactor (gamma): How much to value future rewards (0 to 1)
    //   - 0.95 means "future rewards are almost as good as current rewards"
    //   - 0.5 means "only care about the next few steps"
    //   - High values encourage long-term planning
    //
    // epsilon: How often to take random actions (0 to 1)
    //   - 0.3 = take random action 30% of the time, best action 70% of the time
    //   - High epsilon = more EXPLORATION (trying new things)
    //   - Low epsilon = more EXPLOITATION (using what you know)
    //
    // epsilonMin: Smallest epsilon can be (never go below this)
    // epsilonDecay: How fast epsilon decreases each step (0.995 = very slow decay)
    //
    // memorySize: How many experiences to remember for replay
    //   - Larger memory = more diverse learning experiences
    //   - But uses more computer memory
    //
    
    constructor(config = {}) {
        // Learning parameters (how the agent learns from experience)
        this.learningRate = config.learningRate || 0.1;      // Alpha: trust new info 10%
        this.discountFactor = config.discountFactor || 0.95; // Gamma: value future at 95%
        this.epsilon = config.epsilon || 0.3;               // Explore 30%, Exploit 70%
        this.epsilonMin = config.epsilonMin || 0.05;        // Always explore at least 5%
        this.epsilonDecay = config.epsilonDecay || 0.995;   // Very slow decay rate
        
        // Memory for experience replay (like a diary of past experiences)
        this.memory = [];
        this.memorySize = config.memorySize || 5000;
        
        // Q-table: This is the agent's "brain" - a big lookup table
        // Key: state (like "position 5,3, goal at 2,7")
        // Value: estimated future reward for each action
        // Example: qTable.get("5,3,2,7,0.5,1|0") = 10.5 (moving up is worth 10.5 points)
        this.qTable = new Map();
        
        // Tracking for adaptation (how well is the agent learning?)
        this.recentRewards = [];      // Last 100 rewards for performance monitoring
        this.episodeRewards = [];     // Total reward for each complete episode
        this.totalSteps = 0;          // Total actions taken across all episodes
        this.goalChangeDetected = false;
        
        // Performance tracking
        this.lastAverageReward = 0;
    }
    
    // ------------------------------------------------------------------------
    // ACT - DECIDE WHAT TO DO
    // ------------------------------------------------------------------------
    // This is the agent's "decision making" function.
    // Given what the agent sees (observation), it chooses an action.
    //
    // THE EPSILON-GREEDY STRATEGY:
    // ----------------------------
    // With probability epsilon: take a RANDOM action (EXPLORE)
    // With probability 1-epsilon: take the BEST action (EXPLOIT)
    //
    // EXPLORATION VS EXPLOITATION TRADEOFF:
    // -------------------------------------
    // - EXPLORATION: Trying new things, might find better strategies
    // - EXPLOITATION: Using what you know works, safe but might miss better options
    //
    // ADAPTIVE FEATURE #1:
    // -------------------
    // When the goal moves, we temporarily INCREASE exploration.
    // Why? Because the old strategies might not work anymore!
    // The hill is in a new place, so the agent should try new directions.
    //
    // QUESTION FOR YOU: Why do we only boost epsilon temporarily?
    // ANSWER: Because after the agent finds the new hill, it should go back to exploiting.
    //
    
    act(observation, info = {}) {
        
        // ADAPTIVE BEHAVIOR: When hill moves, explore more!
        if (info.goalMoved) {
            this.goalChangeDetected = true;
            // Boost epsilon by 0.2 (e.g., from 0.3 to 0.5) but never above 0.5
            const boostedEpsilon = Math.min(0.5, this.epsilon + 0.2);
            if (Math.random() < boostedEpsilon) {
                return Math.floor(Math.random() * 5);  // Random action (explore)
            }
        } else {
            this.goalChangeDetected = false;
        }
        
        // Normal epsilon-greedy decision
        if (Math.random() < this.epsilon) {
            // EXPLORE: Take a random action
            return Math.floor(Math.random() * 5);
        } else {
            // EXPLOIT: Take the best action according to Q-table
            return this._getBestAction(observation);
        }
    }
    
    // ------------------------------------------------------------------------
    // GET BEST ACTION - WHAT DOES THE AGENT THINK IS BEST?
    // ------------------------------------------------------------------------
    // Looks at all 5 possible actions and picks the one with the highest Q-value
    //
    // EXAMPLE:
    // --------
    // For a given state, the Q-table might have:
    // - Action 0 (Up):    Q-value = 10.5
    // - Action 1 (Down):  Q-value = 2.3
    // - Action 2 (Left):  Q-value = -1.2
    // - Action 3 (Right): Q-value = 8.7
    // - Action 4 (Stay):  Q-value = 5.0
    // 
    // The agent would choose Action 0 (Up) because 10.5 is highest.
    //
    
    _getBestAction(observation) {
        const stateKey = this._getStateKey(observation);
        let bestAction = 0;
        let bestValue = -Infinity;
        
        // Check all 5 actions to find the highest Q-value
        for (let a = 0; a < 5; a++) {
            const value = this._getQValue(stateKey, a);
            if (value > bestValue) {
                bestValue = value;
                bestAction = a;
            }
        }
        
        return bestAction;
    }
    
    // ------------------------------------------------------------------------
    // TRAIN - LEARN FROM EXPERIENCE
    // ------------------------------------------------------------------------
    // This is where the agent actually LEARNS.
    // After taking an action and receiving a reward, the agent updates its Q-table.
    //
    // THE LEARNING PROCESS:
    // ---------------------
    // 1. Remember this experience (store in memory)
    // 2. Update Q-values using Q-learning formula
    // 3. Track rewards to measure performance
    // 4. Adjust exploration rate based on performance (ADAPTIVE!)
    // 5. Adjust learning rate based on performance trends (ADAPTIVE!)
    //
    // WHAT IS EXPERIENCE REPLAY?
    // --------------------------
    // Instead of learning from each experience once and forgetting it,
    // we store experiences in a "memory" and learn from them multiple times.
    // This is like studying for a test by reviewing old homework.
    // It makes learning more stable and efficient.
    //
    
    train(observation, action, reward, nextObservation, done, info = {}) {
        
        // ----- STEP 1: Remember this experience -----
        this.memory.push({
            observation: observation.slice(),  // Copy the array (don't store reference)
            action: action,
            reward: reward,
            nextObservation: nextObservation.slice(),
            done: done
        });
        
        // Keep memory from getting too big (remove oldest if necessary)
        if (this.memory.length > this.memorySize) {
            this.memory.shift();  // Remove the oldest experience
        }
        
        // ----- STEP 2: Update Q-values using Q-learning -----
        this._qLearningUpdate(observation, action, reward, nextObservation, done);
        
        // ----- STEP 3: Track recent rewards for performance monitoring -----
        this.recentRewards.push(reward);
        if (this.recentRewards.length > 100) {
            this.recentRewards.shift();  // Keep only last 100 rewards
        }
        
        // ----- STEP 4: ADAPTIVE EPSILON DECAY -----
        // Adjust exploration rate based on how well the agent is doing
        this._adaptiveEpsilonDecay();
        
        // ----- STEP 5: ADAPTIVE LEARNING RATE -----
        // Adjust learning rate based on performance trends
        if (this.totalSteps % 500 === 0) {  // Every 500 steps
            this._adjustLearningRate();
        }
        
        this.totalSteps++;
    }
    
    // ------------------------------------------------------------------------
    // Q-LEARNING UPDATE - THE CORE LEARNING ALGORITHM
    // ------------------------------------------------------------------------
    // This implements the Bellman equation, the heart of Q-learning.
    //
    // THE BELLMAN EQUATION EXPLAINED:
    // -------------------------------
    // 
    // Q(s,a) = Q(s,a) + α * [ R + γ * max(Q(s',a')) - Q(s,a) ]
    // 
    // Where:
    // - Q(s,a) = Current estimated value of taking action 'a' in state 's'
    // - α (alpha) = Learning rate (how much to trust new information)
    // - R = Immediate reward received
    // - γ (gamma) = Discount factor (how much to value future rewards)
    // - max(Q(s',a')) = Best possible future reward from new state
    //
    // WHAT THIS MEANS IN PLAIN ENGLISH:
    // ---------------------------------
    // "Update your estimate for this action based on:
    //  1. What you thought before (current Q)
    //  2. What actually happened (reward + best future reward)
    //  3. The difference between them (the error)"
    //
    // EXAMPLE:
    // --------
    // You think moving up is worth 10 points (Q=10)
    // You move up, get 5 points now, and see that best future is 8 points
    // Your new estimate = 10 + 0.1 * (5 + 0.95*8 - 10) 
    //                   = 10 + 0.1 * (5 + 7.6 - 10)
    //                   = 10 + 0.1 * 2.6
    //                   = 10.26 points
    // You slightly increased your estimate because it worked out better than expected.
    //
    
    _qLearningUpdate(observation, action, reward, nextObservation, done) {
        const stateKey = this._getStateKey(observation);
        const nextStateKey = this._getStateKey(nextObservation);
        
        // Current Q-value (what we thought before)
        const currentQ = this._getQValue(stateKey, action);
        
        // Target Q-value (what we learned from experience)
        let maxNextQ = 0;
        if (!done) {
            // Find the best possible action from the new state
            for (let a = 0; a < 5; a++) {
                maxNextQ = Math.max(maxNextQ, this._getQValue(nextStateKey, a));
            }
        }
        
        // Bellman equation: target = reward + discount * bestFutureQ
        const targetQ = reward + this.discountFactor * maxNextQ;
        
        // Update: newQ = oldQ + learningRate * (target - oldQ)
        const newQ = currentQ + this.learningRate * (targetQ - currentQ);
        this._setQValue(stateKey, action, newQ);
    }
    
    // ------------------------------------------------------------------------
    // ADAPTIVE EPSILON DECAY - EXPLORE MORE WHEN LOSING
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
    // ADAPTIVE LEARNING RATE - LEARN FASTER WHEN CONFUSED
    // ------------------------------------------------------------------------
    // Adjusts how quickly the agent learns from new experiences.
    //
    // THE RULES:
    // ----------
    // - If performance is DROPPING: Increase learning rate (learn faster to correct)
    // - If performance is IMPROVING: Decrease learning rate (fine-tune what works)
    //
    // WHY THIS WORKS:
    // --------------
    // When the environment changes and the agent starts losing, it needs to
    // learn the new rules quickly. Increasing learning rate helps with that.
    // Once it's doing well, slower learning helps avoid overreacting to noise.
    //
    
    _adjustLearningRate() {
        if (this.episodeRewards.length < 10) return;
        
        // Calculate average of last 5 episodes
        const last5 = this.episodeRewards.slice(-5).reduce((a,b) => a + b, 0) / 5;
        // Calculate average of previous 5 episodes (episodes 10-6)
        const prev5 = this.episodeRewards.slice(-10, -5).reduce((a,b) => a + b, 0) / 5;
        
        if (last5 < prev5 * 0.8) {
            // Performance dropped by more than 20%: increase learning rate
            this.learningRate = Math.min(0.3, this.learningRate * 1.1);
            console.log(`Learning rate increased to ${this.learningRate.toFixed(3)}`);
        } else if (last5 > prev5 * 1.2 && this.learningRate > 0.05) {
            // Performance improved by more than 20%: decrease learning rate
            this.learningRate *= 0.95;
        }
    }
    
    // ------------------------------------------------------------------------
    // END EPISODE - CLEANUP AFTER EACH GAME
    // ------------------------------------------------------------------------
    // Called when an episode finishes (after 1000 steps or manually stopped)
    // Records the total reward and potentially adjusts future behavior
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
    // STATE KEY - CONVERT OBSERVATION TO A STRING FOR THE Q-TABLE
    // ------------------------------------------------------------------------
    // The observation is 6 continuous numbers (like -0.23, 0.45, 0.12, etc.)
    // We need to convert this to a DISCRETE key for the Q-table.
    //
    // DISCRETIZATION EXPLAINED:
    // -------------------------
    // Imagine you have a ruler from -1 to 1. You draw 8 boxes on it:
    // Box 0: -1.00 to -0.75
    // Box 1: -0.75 to -0.50
    // Box 2: -0.50 to -0.25
    // Box 3: -0.25 to 0.00
    // Box 4: 0.00 to 0.25
    // Box 5: 0.25 to 0.50
    // Box 6: 0.50 to 0.75
    // Box 7: 0.75 to 1.00
    //
    // A value of -0.3 falls into Box 3.
    // A value of 0.6 falls into Box 6.
    //
    // We do this for all 6 observation values, then join them with commas.
    // This creates a unique key for each "region" of the state space.
    //
    // EXAMPLE:
    // --------
    // Observation: [-0.3, 0.6, 0.1, -0.4, 0.5, 1.0]
    // Discretized: [3, 6, 4, 2, 4, 7]
    // State Key: "3,6,4,2,4,7"
    //
    // Total possible states: 8^6 = 262,144 different state keys
    //
    
    _getStateKey(observation) {
        const bins = 8;  // Divide each dimension into 8 buckets
        const keys = [];
        
        for (let i = 0; i < observation.length; i++) {
            // Map from [-1, 1] to [0, bins-1]
            // (+1 shifts from -1..1 to 0..2, /2 normalizes to 0..1, *bins scales)
            const bin = Math.floor((observation[i] + 1) / 2 * bins);
            keys.push(Math.min(bins - 1, Math.max(0, bin)));
        }
        
        return keys.join(',');
    }
    
    // ------------------------------------------------------------------------
    // Q-VALUE STORAGE - READ AND WRITE TO THE Q-TABLE
    // ------------------------------------------------------------------------
    // The Q-table stores values for each (state, action) pair.
    // Key format: "stateKey|action"
    // Example: "3,6,4,2,4,7|0" means "in state 3,6,4,2,4,7, taking action 0"
    //
    
    _getQValue(stateKey, action) {
        const key = `${stateKey}|${action}`;
        return this.qTable.get(key) || 0;  // Default to 0 if not found
    }
    
    _setQValue(stateKey, action, value) {
        const key = `${stateKey}|${action}`;
        this.qTable.set(key, value);
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


// ============================================================================
// PART 3: THE TRAINER CLASS
// ============================================================================
//
// WHAT IS A TRAINER?
// ------------------
// The trainer orchestrates the whole learning process.
// It runs episodes, calls the agent and environment at the right times,
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
        this.episodeData = [];
        this.listeners = [];      // Functions to call on updates
        
        // Performance tracking
        this.rewardHistory = [];
        this.adaptationHistory = [];
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
    //         updateThreeJSVisualization(data.observation);
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
            finalEpsilon: this.agent.epsilon,
            finalQTableSize: this.agent.qTable.size
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
//   let env = new AdaptiveRL.Environment()
//   let agent = new AdaptiveRL.Agent()
//   let trainer = new AdaptiveRL.Trainer(env, agent)
//
// The 'window' object is the global object in browsers.
// Adding properties to it makes them available everywhere.
//

window.AdaptiveRL = {
    Environment: AdaptiveHillEnvironment,
    Agent: AdaptiveDQNAgent,
    Trainer: AdaptiveRLTrainer
};

// Console messages to confirm loading
console.log('Adaptive RL Module Loaded Successfully');
console.log('Available classes:');
console.log('  - AdaptiveRL.Environment  (the game world)');
console.log('  - AdaptiveRL.Agent        (the learning AI)');
console.log('  - AdaptiveRL.Trainer      (runs the training loop)');