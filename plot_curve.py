from utils import plot_multiple_learning_curves

plot_multiple_learning_curves('plots/5-oc-runs.png', 
                              'results/tetris-option-critic-0.txt', 
                              'results/tetris-option-critic-1.txt',
                              'results/tetris-option-critic-2.txt',
                              'results/tetris-option-critic-3.txt',
                              'results/tetris-option-critic-4.txt')