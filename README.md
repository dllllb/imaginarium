# Imaginary world planning

- use Monte-Carlo Tree Search to find effective trajectory in imaginary world based on world model
  - i. e. mark trajectory elements according to the sum of imaginary rewards from the word model
    - use sum of node rewards in bandits algorithm to select the next action for the explored trajectory
- the approach can be compared with Monte-Carlo Tree Search for the real deterministic environment
  - some OpenAI Gym environments are deterministic
- Single-Player Monte-Carlo Tree Search (SP-MCTS) MCST variant can possibly be used instead of UCB based approach
