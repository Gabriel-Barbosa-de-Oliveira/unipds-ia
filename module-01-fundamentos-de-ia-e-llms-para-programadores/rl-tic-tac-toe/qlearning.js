// In game.js

// Q-Learning Agent with localStorage support
class QLearning {
    constructor(lr = 0.1, gamma = 0.9, epsilon = 0.1) {
        this.q = new Map(); // Stores Q-values: { state => [q_action_0, q_action_1, ...] }
        this.lr = lr; // Learning Rate (α)
        this.gamma = gamma; // Discount Factor (γ)
        this.epsilon = epsilon; // Exploration Rate (ε)
        this.difficulty = 'intermediate';
    }

    //Return the current board state
    getQ(state) {
        if (!this.q.has(state)) {
            this.q.set(state, Array(9).fill(0));
        }
        return this.q.get(state);
    }

    //Initializes the epsilon greedy strategy
    //The logic first checks the difficulty. 
    //'Beginner' is mostly random, while 'Expert' defers to a separate
    //perfect-play algorithm.
    getAction(state, available) {
        // Difficulty-based behavior
        if (this.difficulty === 'beginner') {
            // 70% random moves for beginner
            if (Math.random() < 0.7) {
                return available[~~(Math.random() * available.length)];
            }
        } else if (this.difficulty === 'expert') {
            // Use minimax for perfect play
            return this.getMinimaxAction(state, available);
        }

        // Intermediate: epsilon-greedy
        if (Math.random() < this.epsilon) {
            return available[~~(Math.random() * available.length)];
        }
        const q = this.getQ(state);
        return available.reduce((best, a) => q[a] > q[best] ? a : best, available[0]);
    }

    //This method is where the learning rule is at
    //Q(s, a) ← Q(s, a) + α [r + γ max(a') Q(s', a') − Q(s, a)]

    update(s, a, r, s2, available2) {
        const q = this.getQ(s);
        //maxQ2 calculates the max Q(s',a') part of the formula – the best possible Q-value the AI can get from its next move.
        const maxQ2 = available2.length ? Math.max(...available2.map(a_prime => this.getQ(s2)[a_prime])) : 0;
        q[a] += this.lr * (r + this.gamma * maxQ2 - q[a]);
    }

    //For our 'Expert' level, we'll implement the minimax algorithm
    //a classic recursive algorithm from game theory that guarantees perfect play.
    getMinimaxAction(state, available) {
        let bestScore = -Infinity;
        let bestMove = available[0];

        for (const move of available) {
            const newState = state.substring(0, move) + 'O' + state.substring(move + 1);
            const score = this.minimax(newState, 0, false);
            if (score > bestScore) {
                bestScore = score;
                bestMove = move;
            }
        }
        return bestMove;
    }

    minimax(state, depth, isMaximizing) {
        const winner = this.checkWinnerStatic(state);
        if (winner === 'O') return 10 - depth;
        if (winner === 'X') return depth - 10;
        if (winner === 'draw') return 0;

        const available = [...state].map((c, i) => c === '-' ? i : null).filter(x => x !== null);

        if (isMaximizing) {
            let best = -Infinity;
            for (const move of available) {
                const newState = state.substring(0, move) + 'O' + state.substring(move + 1);
                best = Math.max(best, this.minimax(newState, depth + 1, false));
            }
            return best;
        } else {
            let best = Infinity;
            for (const move of available) {
                const newState = state.substring(0, move) + 'X' + state.substring(move + 1);
                best = Math.min(best, this.minimax(newState, depth + 1, true));
            }
            return best;
        }
    }

    checkWinnerStatic(state) {
        const patterns = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]];
        for (const p of patterns) {
            if (state[p[0]] !== '-' && state[p[0]] === state[p[1]] && state[p[1]] === state[p[2]]) {
                return state[p[0]];
            }
        }
        return state.includes('-') ? null : 'draw';
    }

    //methods for epsilon decay, resetting the AI's memory,
    //and saving/loading the Q-table to localStorage.

    decay() {
        this.epsilon = Math.max(0.01, this.epsilon * 0.995);
    }

    reset() {
        this.q.clear();
        this.epsilon = 0.1;
    }

    save() {
        const data = {
            q: Array.from(this.q.entries()),
            lr: this.lr,
            gamma: this.gamma,
            epsilon: this.epsilon,
            difficulty: this.difficulty
        };
        localStorage.setItem('tictactoe_ai', JSON.stringify(data));
    }

    load() {
        const saved = localStorage.getItem('tictactoe_ai');
        if (!saved) return false;

        try {
            const data = JSON.parse(saved);
            this.q = new Map(data.q);
            this.lr = data.lr;
            this.gamma = data.gamma;
            this.epsilon = data.epsilon;
            this.difficulty = data.difficulty || 'intermediate';
            return true;
        } catch (e) {
            console.error('Failed to load AI state:', e);
            return false;
        }
    }

    clearStorage() {
        localStorage.removeItem('tictactoe_ai');
    }

}