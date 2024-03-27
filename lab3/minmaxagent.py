class MinMaxAgent:
    def __init__(self, game, depth=4, my_token='o'):
        self.game = game
        self.depth = depth
        self.my_token = my_token

    def decide(self, game_state):
        return self.find_best_move(game_state)

    def evaluate(self, state):
        score = 0
        center_column = [row[state.width // 2] for row in state.board]
        center_count = center_column.count(self.my_token)
        score += center_count * 3  # Arbitrary weight for tokens in the center column

        # Function to count sequences of tokens
        def count_sequences(board, token, sequence_length):
            count = 0
            # Horizontal
            for row in board:
                for column in range(state.width - 3):
                    if row[column:column + sequence_length].count(token) == sequence_length:
                        count += 1
            # Vertical
            for column in range(state.width):
                for row in range(state.height - 3):
                    if all(board[row + i][column] == token for i in range(sequence_length)):
                        count += 1
            # Positive Diagonal
            for row in range(state.height - 3):
                for column in range(state.width - 3):
                    if all(board[row + i][column + i] == token for i in range(sequence_length)):
                        count += 1
            # Negative Diagonal
            for row in range(3, state.height):
                for column in range(state.width - 3):
                    if all(board[row - i][column + i] == token for i in range(sequence_length)):
                        count += 1
            return count

        score += count_sequences(state.board, self.my_token, 4) * 100000  # Winning condition
        score -= count_sequences(state.board, 'x', 4) * 100000  # Opponent's winning condition

        score += count_sequences(state.board, self.my_token, 3) * 10
        score -= count_sequences(state.board, 'x', 3) * 10

        return score

    def minimax(self, state, depth, alpha, beta, maximizing_player):
        if depth == 0 or state.game_over:
            return self.evaluate(state)

        if maximizing_player:
            max_eval = float('-inf')
            for move in state.possible_drops():
                state_copy = state.clone()
                state_copy.drop_token(move)
                eval = self.minimax(state_copy, depth-1, alpha, beta, False)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in state.possible_drops():
                state_copy = state.clone()
                state_copy.drop_token(move)
                eval = self.minimax(state_copy, depth-1, alpha, beta, True)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

    def find_best_move(self, game_state):
        best_move = None
        best_value = float('-inf')
        for move in game_state.possible_drops():
            state_copy = game_state.clone()  # Use the passed game_state instead of self.game
            state_copy.drop_token(move)
            move_value = self.minimax(state_copy, self.depth, float('-inf'), float('inf'), False)
            if move_value > best_value:
                best_value = move_value
                best_move = move
        return best_move
