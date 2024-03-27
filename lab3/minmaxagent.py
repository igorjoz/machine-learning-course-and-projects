class MinMaxAgent:
    def __init__(self, game, depth=5, my_token='o'):
        self.game = game  # This might still be needed for other parts of the class
        self.depth = depth
        self.my_token = my_token

    def decide(self, game_state):
        """
        Decides the best move based on the current game state.

        Parameters:
        - game_state: The current state of the Connect 4 game.

        Returns:
        - An integer representing the column number into which to drop the token.
        """
        # Assuming find_best_move is implemented to use the Minimax algorithm
        # and returns the best column for the next move.
        return self.find_best_move(game_state)

        # Adjust the signature of find_best_move to accept the game state
        # def find_best_move(self, game_state):
        #     """
        #     Finds the best move (column) to play next given the current game state.
        #
        #     Parameters:
        #     - game_state: The current state of the game.
        #
        #     Returns:
        #     - An integer representing the best column to play next.
        #     """
        #     # Implementation of finding the best move goes here
        #     # This will likely involve calling the minimax method with the current state
        #     best_column = None
        #     # Your logic to determine the best column
        #     return best_column

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

        # Potential 3-in-a-row sequences for the agent and the opponent
        score += count_sequences(state.board, self.my_token, 3) * 10
        score -= count_sequences(state.board, 'x', 3) * 10

        return score

    def minimax(self, state, depth, alpha, beta, maximizingPlayer):
        if depth == 0 or state.game_over:
            return self.evaluate(state)

        if maximizingPlayer:
            maxEval = float('-inf')
            for move in state.possible_drops():
                state_copy = state.clone()  # Assuming a method to clone the current state
                state_copy.drop_token(move)
                eval = self.minimax(state_copy, depth-1, alpha, beta, False)
                maxEval = max(maxEval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return maxEval
        else:
            minEval = float('inf')
            for move in state.possible_drops():
                state_copy = state.clone()
                state_copy.drop_token(move)
                eval = self.minimax(state_copy, depth-1, alpha, beta, True)
                minEval = min(minEval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return minEval

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
