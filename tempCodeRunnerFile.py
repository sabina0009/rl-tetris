x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, f'plots/{filename}.png')