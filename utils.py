import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)

def plot_multiple_learning_curves(figure_file, *results_files):
    colours = ['b','g', 'r', 'c', 'm', 'y', 'k']
    j=0
    for file in results_files:
        scores = np.loadtxt(file, dtype=int)
        running_avg = np.zeros(len(scores))
        for i in range(len(running_avg)):
            running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
        x = [n for n in range(len(scores))]
        plt.plot(x, running_avg, color=colours[j], label=f'{file[8:-4]}')
        plt.title('Running average of previous 100 scores')
        plt.legend()
        j += 1
    plt.savefig(figure_file)

def plot_average_learning_curve(filename, runs):
    results_files = [f'results/{filename}/{filename}-{i}.txt' for i in range(runs)]
    scores = []
    for i in range(runs):
        score_history = np.loadtxt(results_files[i], dtype=int)
        scores.append(score_history)
    run_lengths = [len(score_history) for score_history in scores]
    min_len = min(run_lengths)
    x = [n for n in range(min_len)]
    avg_scores = []
    for i in range(min_len):
        avg = sum([scores[j][i] for j in range(runs)]) / runs
        avg_scores.append(avg)
    np.savetxt(f'results/average.txt', avg_scores, fmt='%d')
    running_avg = np.zeros(len(avg_scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(avg_scores[max(0, i-100):(i+1)])
    print(f'...plotting average or {runs} runs')
    plt.plot(x, running_avg)
    plt.title(f'{filename} average of {runs} runs')
    plt.savefig(f'plots/{filename}/average.png')