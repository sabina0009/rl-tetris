from utils import plot_multiple_learning_curves, plot_average_learning_curve

#plot_average_learning_curve('tetris-option-critic', 5)

plot_multiple_learning_curves('plots/2-vs-4-options.png',
                              'results/tetris-option-critic-average.txt',
                              'results/tetris-options=4-average.txt',
)