from utils import plot_multiple_learning_curves

plot_multiple_learning_curves('plots/tetris-eta.png', 
            'results/tetris-optioncritic-entropyreg.txt',
            'results/tetris-optioncritic-eta=0.1.txt',
            'results/tetris-optioncritic-eta=0.05.txt')
