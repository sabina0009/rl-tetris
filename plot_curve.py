from utils import plot_multiple_learning_curves, plot_average_learning_curve

#plot_average_learning_curve('tetris-options=2-8x4', 5)


plot_multiple_learning_curves('plots/all-aocs.png',
    'results/tetris20x10-aoc-2options-200000steps/tetris20x10-aoc-2options-200000steps-average.txt',
    'results/tetris20x10-aoc-2options-piece_height-200000steps/tetris20x10-aoc-2options-piece_height-200000steps-average.txt',
    'results/tetris20x10-aoc-4options-200000steps/tetris20x10-aoc-4options-200000steps-average.txt',
    'results/tetris20x10-attentionoc-2options-200000steps/tetris20x10-attentionoc-2options-200000steps-average.txt'       
)