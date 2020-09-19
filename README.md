# Imaginary world planning

Use Monte-Carlo Tree Search to find effective trajectory in imaginary world based on world model. I. e. mark trajectory elements according to the sum of imaginary rewards from the word model. Use sum of node rewards in bandits algorithm to select the next action for the explored trajectory.

The approach can be compared with Monte-Carlo Tree Search for the real deterministic environment. Some OpenAI Gym environments are deterministic.

Single-Player Monte-Carlo Tree Search (SP-MCTS) MCST variant can possibly be used instead of UCB based approach

## Related work

### Monte-carlo search for Atari

See paper: Deep Learning for Real-Time Atari Game Play Using Offline Monte-Carlo Tree Search Planning / Guo et al. / NIPS 2014

**Algorithm:**
1. Use MCTS from the initial state to learn good action distributions for a set of states
2. Train a CNN to mimic MCST action distribtions by trainging for multinominal classification
