class AlphaBetaAgent:
    def __init__(self, my_token, depth=5):
        self.my_token = my_token
        self.depth = depth

    def evaluate(self, state):
        score = 0
        center_column = [row[state.width // 2] for row in state.board]
        center_count = center_column.count(self.my_token)
        score += center_count * 3  # Center column advantage

        # Helper function to count sequences
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
            # Diagonals
            for row in range(state.height - 3):
                for column in range(state.width - 3):
                    if all(board[row + i][column + i] == token for i in range(sequence_length)):
                        count += 1
                    if all(board[row + i][state.width - 1 - column - i] == token for i in range(sequence_length)):
                        count += 1
            return count

        # Score sequences for the agent
        score += count_sequences(state.board, self.my_token, 4) * 100000
        score += count_sequences(state.board, self.my_token, 3) * 10
        score += count_sequences(state.board, self.my_token, 2) * 1

        # Score sequences against the agent
        opponent_token = 'x' if self.my_token == 'o' else 'o'
        score -= count_sequences(state.board, opponent_token, 4) * 100000
        score -= count_sequences(state.board, opponent_token, 3) * 10
        score -= count_sequences(state.board, opponent_token, 2) * 1

        return score

    def alphabeta(self, state, depth, alpha, beta, maximizing_player):
        if depth == 0 or state.game_over:
            return self.evaluate(state)

        if maximizing_player:
            value = float('-inf')
            for move in state.possible_drops():
                state_copy = state.clone()
                state_copy.drop_token(move)
                value = max(value, self.alphabeta(state_copy, depth-1, alpha, beta, False))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break  # Beta cut-off
            return value
        else:
            value = float('inf')
            for move in state.possible_drops():
                state_copy = state.clone()
                state_copy.drop_token(move)
                value = min(value, self.alphabeta(state_copy, depth-1, alpha, beta, True))
                beta = min(beta, value)
                if beta <= alpha:
                    break  # Alpha cut-off
            return value

    def find_best_move(self, game_state, maximizing_player=True):
        best_move = None
        best_value = float('-inf') if maximizing_player else float('inf')

        for move in game_state.possible_drops():
            state_copy = game_state.clone()
            state_copy.drop_token(move)
            value = self.alphabeta(state_copy, self.depth, float('-inf'), float('inf'), not maximizing_player)

            if maximizing_player and value > best_value:
                best_value = value
                best_move = move
            elif not maximizing_player and value < best_value:
                best_value = value
                best_move = move

        return best_move

    def decide(self, game_state):
        return self.find_best_move(game_state, self.my_token == 'o')  # Assuming 'o' is maximizing
