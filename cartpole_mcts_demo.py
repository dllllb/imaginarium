import gaming
from gym_game import GymNStepsGame
from mcts import MCTS

ttt = GymNStepsGame('CartPole-v1')
s1 = MCTS(ttt, n_plays=50, max_depth=50, player=1)

ttt_vis = GymNStepsGame('CartPole-v1', render=True)
state, rewards, steps, log = gaming.play_game(ttt_vis, [s1], max_turns=500)
print()
print(f'steps: {steps}')
print(rewards)
print([a for p, a in log])
