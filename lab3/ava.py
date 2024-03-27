from exceptions import GameplayException
from connect4 import Connect4
from randomagent import RandomAgent
from alphabetaagent import AlphaBetaAgent

connect4 = Connect4(width=7, height=6)

agent1 = RandomAgent('o')

# agent1 = AlphaBetaAgent(my_token='o', depth=5)
# agent2 = AlphaBetaAgent(my_token='x', depth=5)

# agent1 = AlphaBetaAgent(my_token='o', depth=4)
agent2 = AlphaBetaAgent(my_token='x', depth=4)

while not connect4.game_over:
    connect4.draw()
    try:
        if connect4.who_moves == agent1.my_token:
            n_column = agent1.decide(connect4)
        else:
            # n_column = agent2.decide(connect4)
            n_column = agent2.find_best_move(connect4, maximizing_player=True)
        connect4.drop_token(n_column)
    except (ValueError, GameplayException):
        print('invalid move')

connect4.draw()
