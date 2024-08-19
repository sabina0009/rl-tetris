from utils import plot_multiple_learning_curves

plot_multiple_learning_curves('plots/dqn-target-and-policy-nets.png', 
                              'results/tetris-dqn-target-net.txt', 
                              'results/tetris-dqn.txt')
