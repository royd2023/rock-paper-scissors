import numpy as np

# Map R/P/S to numbers and back
move_to_index = {"R": 0, "P": 1, "S": 2}
index_to_move = {0: "R", 1: "P", 2: "S"}

class QLearningAgent:
    def __init__(self):
        self.q_table = np.zeros((81, 3)) + 0.5
        self.learning_rate = 0.15
        self.discount = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995

        self.last_state = 0
        self.last_action = 0
        self.first_round = True

        self.my_history = []
        self.opponent_history = []
        self.my_sequence_tracker = {}
        
        # Abbey counter-strategy
        self.abbey_counter_active = False
        self.recent_results = []
        
        # Kris detection and counter
        self.kris_detected = False
        self.kris_detection_window = []
        self.early_kris_check = []  # For faster detection in first few rounds

    def track_my_sequences(self):
        """Track our own 2-move sequences exactly like Abbey tracks them"""
        if len(self.my_history) >= 2:
            last_two = "".join([index_to_move[self.my_history[-2]], index_to_move[self.my_history[-1]]])
            if last_two not in self.my_sequence_tracker:
                self.my_sequence_tracker[last_two] = {"R": 0, "P": 0, "S": 0}

    def detect_kris_pattern(self, opponent_move):
        """Detect if opponent is playing like Kris (always countering our last move)"""
        # Kris responds to OUR previous move
        if len(self.my_history) >= 2 and opponent_move is not None:
            # Look at what we played in the previous round
            our_previous_move = self.my_history[-2]
            # Kris should play what beats that move
            expected_kris_move = (our_previous_move + 1) % 3
            
            is_kris_move = (opponent_move == expected_kris_move)
            self.kris_detection_window.append(is_kris_move)
            
            # Early detection for first few rounds
            if len(self.my_history) <= 6:
                self.early_kris_check.append(is_kris_move)
                if len(self.early_kris_check) >= 3 and sum(self.early_kris_check) >= 3:
                    self.kris_detected = True
            
            # Keep only last 8 moves for detection
            if len(self.kris_detection_window) > 8:
                self.kris_detection_window.pop(0)
            
            # Detect Kris if 6+ out of last 8 moves match Kris pattern
            if len(self.kris_detection_window) >= 6:
                kris_matches = sum(self.kris_detection_window)
                if kris_matches >= 6:
                    self.kris_detected = True
                elif kris_matches <= 3:
                    self.kris_detected = False

    def get_anti_kris_move(self):
        """Get a move designed to beat Kris's strategy"""
        if len(self.my_history) == 0:
            return np.random.choice(3)
        
        # Kris will play what beats our last move
        # So if we played R (0), Kris plays P (1)
        # We need to play S (2) to beat Kris's P
        our_last_move = self.my_history[-1]
        kris_expected_move = (our_last_move + 1) % 3
        anti_kris_move = (kris_expected_move + 1) % 3
        
        # Add some randomness to avoid being too predictable
        if np.random.random() < 0.15:
            return np.random.choice(3)
        
        return anti_kris_move

    def abbey_would_predict(self):
        """Predict what Abbey would predict we'll play next"""
        if len(self.my_history) < 2:
            return None
        
        # Simulate Abbey's logic
        my_last_move = index_to_move[self.my_history[-1]]
        
        # Abbey looks at: our_last_move + "R", our_last_move + "P", our_last_move + "S"
        potential_plays = [my_last_move + "R", my_last_move + "P", my_last_move + "S"]
        
        # Find which one we've done most often historically
        max_count = -1
        abbey_prediction = None
        
        for play_seq in potential_plays:
            count = self.my_sequence_tracker.get(play_seq, {"R": 0, "P": 0, "S": 0})
            total_count = sum(count.values())
            if total_count > max_count:
                max_count = total_count
                abbey_prediction = play_seq[-1]  # Last character is the predicted move
        
        return abbey_prediction

    def update_sequence_tracker(self, move):
        """Update our sequence tracker when we make a move"""
        if len(self.my_history) >= 2:
            last_two = "".join([index_to_move[self.my_history[-2]], index_to_move[self.my_history[-1]]])
            move_char = index_to_move[move]
            
            if last_two not in self.my_sequence_tracker:
                self.my_sequence_tracker[last_two] = {"R": 0, "P": 0, "S": 0}
            
            self.my_sequence_tracker[last_two][move_char] += 1

    def should_counter_abbey(self):
        """Decide if we should use Abbey-specific counter strategy"""
        # Don't use Abbey counter if we detected Kris
        if self.kris_detected:
            return False
            
        # Activate Abbey counter if we're performing poorly
        if len(self.recent_results) >= 20:
            recent_wins = sum(1 for r in self.recent_results[-20:] if r == 1)
            win_rate = recent_wins / 20
            
            if win_rate < 0.55:  # If win rate drops below 55%
                return True
        
        return False

    def get_abbey_counter_move(self):
        """Get a move specifically designed to counter Abbey's prediction"""
        abbey_prediction = self.abbey_would_predict()
        
        if abbey_prediction is None:
            return None
        
        # Abbey will play what beats her prediction of our move
        # So if Abbey predicts we'll play R, she'll play P (to beat R)
        # We need to play S (to beat her P)
        abbey_predicted_move = move_to_index[abbey_prediction]
        abbey_counter_to_prediction = (abbey_predicted_move + 1) % 3  # What Abbey will play
        our_counter_to_abbey = (abbey_counter_to_prediction + 1) % 3  # What beats Abbey's move
        
        return our_counter_to_abbey

    def get_state(self, prev_play):
        if prev_play == "" or prev_play not in move_to_index:
            current_move = 0
        else:
            current_move = move_to_index[prev_play]

        if not hasattr(self, "last_four_moves"):
            self.last_four_moves = [0, 0, 0, 0]

        self.last_four_moves.pop(0)
        self.last_four_moves.append(current_move)

        combined_state = (
            self.last_four_moves[0] * 27 + 
            self.last_four_moves[1] * 9 + 
            self.last_four_moves[2] * 3 + 
            self.last_four_moves[3]
        )
        
        return combined_state

    def choose_move(self, state):
        # Track our sequences
        self.track_my_sequences()
        
        # Priority 1: Counter Kris if detected
        if self.kris_detected:
            return self.get_anti_kris_move()
        
        # Priority 2: Check if we should use Abbey counter-strategy
        if self.should_counter_abbey():
            abbey_counter_move = self.get_abbey_counter_move()
            if abbey_counter_move is not None and np.random.random() < 0.7:
                return abbey_counter_move
        
        # Priority 3: Break predictable patterns to confuse pattern-based opponents
        if len(self.my_history) >= 4:
            # Check if we're creating predictable patterns
            last_four = self.my_history[-4:]
            if len(set(last_four)) == 1:  # All same move
                return np.random.choice(3)
            
            # Check for alternating patterns
            if len(last_four) == 4 and last_four[0] == last_four[2] and last_four[1] == last_four[3]:
                return np.random.choice(3)
        
        # Priority 4: Standard Q-learning
        if np.random.rand() < self.epsilon:
            return np.random.choice(3)
        return np.argmax(self.q_table[state])

    def update_q_table(self, reward, new_state):
        if self.last_state >= len(self.q_table) or new_state >= len(self.q_table):
            return
            
        old = self.q_table[self.last_state, self.last_action]
        future = np.max(self.q_table[new_state])
        self.q_table[self.last_state, self.last_action] = old + self.learning_rate * (reward + self.discount * future - old)
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)


agent = QLearningAgent()

def player(prev_play, history=[]):
    history.append(prev_play)

    state = agent.get_state(prev_play)

    if prev_play in move_to_index:
        opponent_move = move_to_index[prev_play]
    else:
        opponent_move = None

    if agent.first_round:
        move = agent.choose_move(state)
        agent.last_state = state
        agent.last_action = move
        agent.first_round = False
        agent.my_history.append(move)
        if opponent_move is not None:
            agent.opponent_history.append(opponent_move)
        return index_to_move[move]

    # Store opponent's move for pattern detection
    if opponent_move is not None:
        agent.opponent_history.append(opponent_move)
        agent.detect_kris_pattern(opponent_move)

    # Calculate result
    us = agent.last_action
    result = (us - opponent_move) % 3 if opponent_move is not None else 0

    if result == 1:
        reward = 1
        result_code = 1
    elif result == 0:
        reward = 0
        result_code = 0
    else:
        reward = -1
        result_code = -1

    # Track results for Abbey counter-strategy activation
    agent.recent_results.append(result_code)
    if len(agent.recent_results) > 30:
        agent.recent_results.pop(0)

    agent.update_q_table(reward, state)

    move = agent.choose_move(state)
    agent.last_state = state
    agent.last_action = move
    
    # Update our sequence tracker AFTER choosing the move
    agent.update_sequence_tracker(move)
    agent.my_history.append(move)
    
    return index_to_move[move]
