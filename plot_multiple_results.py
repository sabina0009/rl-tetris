from utils import plot_multiple_learning_curves

plot_multiple_learning_curves('plots/shared-vs-separate', 
                              'results/PPO separate nets/tetris-PPO-average.txt', 
                              'results/PPO shared nets/tetris-PPO-average.txt')
