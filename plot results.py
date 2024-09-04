import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os 

colours = ['lightcoral', 'gold', 'palegoldenrod', 'yellowgreen', 
           'mediumaquamarine', 'turquoise', 'deepskyblue', 'cornflowerblue', 'mediumslateblue',
           'plum', 'deeppink', 'lightpink']

def plot_curve(results_file_path, colour, name, running_average=False, truncate=None):
    scores = np.loadtxt(results_file_path)
    if truncate:
        scores = scores[:truncate]
    x = [i for i in range(len(scores))]
    if running_average:
        y = np.zeros(len(scores))
        for i in range(len(y)):
            y[i] = np.mean(scores[max(0, i-100):(i+1)])
            title = 'Running average of previous 100 scores'
    else:
        y = scores
        title = 'Sum of rewards per episode'
    plt.plot(x, y, color = colour, label=name)
    return title

def plot_learning_curve(board_size, model, options, running_average=False):
    if board_size == '8 x 4':
        running_average = True
    if options == None:
        results_file_path = f'example results/scores/{board_size}/{model}/average.txt'
    else:
        results_file_path = f'example results/scores/{board_size}/{model}/{options} options/average.txt'
    colour = np.random.choice(colours)
    if not os.path.isdir(f'example results/plots/{board_size}/{model}'):
        os.mkdir(f'example results/plots/{board_size}/{model}')
    dir = f'example results/plots/{board_size}/{model}'
    if options == None:
        name = f'{model} average'
    else:
        name = f'{model} - {options} options average'
    title = plot_curve(results_file_path, colour, name, running_average)
    plt.title(title)
    plt.grid()
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards')
    plt.savefig(f'{dir}/{name}', dpi=300)
    plt.clf()

def plot_multiple_options(board_size, model, running_average=True):
    if board_size == '8 x 4':
        running_average = True
    results_file_paths = [f'example results/scores/{board_size}/{model}/{i} options/average.txt' for i in [2, 4, 8]]
    colors = np.random.choice(colours, 3, replace=False)
    for i in range(len(results_file_paths)):
        label = f'{2**(i+1)} options'
        title = plot_curve(results_file_paths[i], colors[i], label, running_average)
    plt.title(title)
    plt.grid()
    plt.legend()
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards')
    plt.savefig(f'example results/plots/Tetris {board_size}-{model}-2, 4 and 8 options.png', dpi=300)
    plt.clf()

def plot_multiple_runs(filename, board_size, model, options, running_average=False):
    if board_size == '8 x 4':
        running_average = True
    if options == None:
        results_file_paths = [f'example results/scores/{board_size}/{model}/{filename}-{i}.txt' for i in range(5)]
    else:
        results_file_paths = [f'example results/scores/{board_size}/{model}/{options} options/{filename}-{i}.txt' for i in range(5)]
    colors = np.random.choice(colours, 5, replace=False)
    for i in range(len(results_file_paths)):
        label = f'Run {i}'
        title = plot_curve(results_file_paths[i], colors[i], label, running_average)
    plt.title(title)
    plt.grid()
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards')
    if not os.path.isdir(f'example results/plots/{board_size}/{model}'):
        os.mkdir(f'example results/plots/{board_size}/{model}')
    dir = f'example results/plots/{board_size}/{model}'
    if options and not os.path.isdir(f'example results/plots/{board_size}/{model}/{options} options'):
        os.mkdir(f'example results/plots/{board_size}/{model}/{options} options')
        dir = f'example results/plots/{board_size}/{model}/{options} options'
    if options == None:
        plt.savefig(f'{dir}/{model}-5 runs.png', dpi=300)
    else:
        plt.savefig(f'{dir}/{model}-{options} options-5 runs.png', dpi=300)
    plt.clf()

def plot_multiple_models(*models, board_size, options=None, running_average=True, max_eps=None):
    results_file_paths = []
    labels = []
    for model in models:
        if options == 'all' and model in OC_models:
            results_file_paths.append(f'example results/scores/{board_size}/{model}/2 options/average.txt')
            results_file_paths.append(f'example results/scores/{board_size}/{model}/4 options/average.txt')
            results_file_paths.append(f'example results/scores/{board_size}/{model}/8 options/average.txt')
            labels.append(f'{model} - 2 options')
            labels.append(f'{model} - 4 options')
            labels.append(f'{model} - 8 options')
        elif options and model in OC_models:
            results_file_paths.append(f'example results/scores/{board_size}/{model}/{options} options/average.txt')
            labels.append(f'{model} - {options} options')
        else:
            results_file_paths.append(f'example results/scores/{board_size}/{model}/average.txt')
            labels.append(model)
    colors = np.random.choice(colours, len(results_file_paths), replace=False)
    for i in range(len(results_file_paths)):
        if not max_eps:
            title = plot_curve(results_file_paths[i], colors[i], labels[i], running_average)
        else:
            title = plot_curve(results_file_paths[i], colors[i], labels[i], running_average, truncate=max_eps)
    plt.title(title)
    plt.grid()
    plt.legend()
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards')
    name = models[0]
    for i in range(1, len(models)):
        name += f' vs {models[i]}'
    if options:
        name += f' {options} options'
    plt.savefig(f'example results/plots/Tetris {board_size}-{name}.png', dpi=300)
    plt.clf()

def plot_any(results_file_paths, plot_name, max_eps=None):
    colors = np.random.choice(colours, len(results_file_paths), replace=False)
    names = []
    for i in range(len(results_file_paths)):
        path = results_file_paths[i]
        info = path.split('/')
        board_size = info[1]
        model = info[2]
        name = f'{model}'
        if len(info) == 5:
            options = info[3]
            name += f'-{options}'
        title = plot_curve(path, colors[i], name, running_average=True, truncate=max_eps)
        names.append(name)
    plt.title(title)
    plt.grid()
    plt.legend()
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards')
    plt.savefig(f'example results/plots/Tetris {board_size}-{plot_name}.png', dpi = 300)
    plt.clf()

def plot_lines_cleared(board_size, model, filename, options=None, running_average=False):
    if board_size == '8 x 4':
        running_average = True
    if options == None:
        results_file_path = f'example results/scores/{board_size}/{model}/lines_cleared'
    else:
        results_file_path = f'example results/scores/{board_size}/{model}/{options} options/lines_cleared'
    find_average(results_file_path, filename)
    results_file_path += f'/average.txt'
    colour = np.random.choice(colours)
    if not os.path.isdir(f'plots/{board_size}/{model}'):
        os.mkdir(f'plots/{board_size}/{model}')
    dir = f'example results/plots/{board_size}/{model}'
    if options == None:
        name = f'{model} - lines cleared'
    else:
        name = f'{model} - {options} options - lines cleared'
    title = plot_curve(results_file_path, colour, name, running_average)
    plt.title(title)
    plt.grid()
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards')
    plt.savefig(f'{dir}/{name}', dpi=300)
    plt.clf()

def plot_other_envs(filenames, plot_name, max_eps = None):
    running_average = True
    file_paths = []
    colors = np.random.choice(colours, len(filenames), replace=False)
    for i in range(len(filenames)):
        file_paths.append(f'example results/scores/Other envs/{filenames[i]}/average.txt')
        info = filenames[i].split()
        if info[2] in ['OC', 'PPOC']:
            name = f'{info[2]} {info[4]} {info[5]}'
        else:
            name = info[2]
        title = plot_curve(file_paths[i], colors[i], name, running_average, max_eps)
    plt.title(title)
    plt.grid()
    plt.legend()
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards')
    plt.savefig(f'plots/{plot_name}', dpi=300)
    plt.clf()
    
def find_average(file_path, filename):
    results_files = [f'{file_path}/{filename}-{i}.txt' for i in range(5)]
    scores = []
    for i in range(5):
        score_history = np.loadtxt(results_files[i], dtype=int)
        scores.append(score_history)
    run_lengths = [len(score_history) for score_history in scores]
    min_len = min(run_lengths)
    avg_scores = []
    for i in range(min_len):
        avg = sum([scores[j][i] for j in range(5)]) / 5
        avg_scores.append(avg)
    print('... saving average ...')
    np.savetxt(f'{file_path}/average.txt', avg_scores, fmt='%d')

#find_average('PPO N=20/results files', 'tetris-PPO')

OC_models = ['Option-critic', 'PPOC', 'Attention OC']
other_models = ['DQN', 'DQN target networks - soft update', 'DQN target networks', 'PPO']
extra_models = ['PPO N=20', 'PPO shared AC networks N=20', 'DQN target networks - 200k steps']

# for model in OC_models:
#     for options in [2, 4]:
#         plot_learning_curve('20 x 10', model, options)
#         plot_learning_curve('8 x 4', model, options)
#     if model != 'Attention OC':
#         plot_learning_curve('20 x 10', model, 8)
#         plot_learning_curve('8 x 4', model, 8)

# for options in [2, 4]:
#     plot_learning_curve('20 x 10', 'Attention OC', options)
#     plot_learning_curve('8 x 4', 'Attention OC', options)

# for model in other_models:
#     plot_learning_curve('20 x 10', model, options=None)
#     plot_learning_curve('8 x 4', model, options=None)

# for model in extra_models:
#     plot_learning_curve('20 x 10', model, options=None)

# for model in OC_models:
#     if model != 'Attention OC':
#         plot_multiple_options('20 x 10', model)
#         plot_multiple_options('8 x 4', model)

# plot_any(['example results/scores/20 x 10/Attention OC/2 options/average.txt',
#           'example results/scores/20 x 10/Attention OC/4 options/average.txt'],
#           'Attention OC-all options')

# plot_any(['example results/scores/8 x 4/Attention OC/2 options/average.txt',
#           'example results/scores/8 x 4/Attention OC/4 options/average.txt'],
#           'Attention OC-all options')

# plot_multiple_models('Option-critic', 'PPOC', board_size='20 x 10', options='all', max_eps=1500)
# plot_multiple_models('Option-critic', 'PPOC', board_size='8 x 4', options='all', max_eps=8000)

# plot_multiple_models('PPO', 'PPO N=20', board_size='20 x 10', options=None, max_eps=2500)
# plot_multiple_models('PPO N=20', 'PPO shared AC networks N=20', board_size='20 x 10', options=None)
# plot_multiple_models('DQN', 'DQN target networks', 'DQN target networks - soft update', board_size='20 x 10', options=None)
# plot_multiple_models('DQN', 'DQN target networks', 'DQN target networks - soft update', board_size='8 x 4', options=None)

# plot_multiple_models('DQN target networks - 200k steps', 'PPO', board_size='20 x 10', max_eps=1500)
# plot_multiple_models('DQN target networks - 300k steps', 'PPO', board_size='8 x 4', options=None, max_eps=7500)

# plot_multiple_models('DQN target networks - 200k steps', 'Option-critic', board_size='20 x 10', options=4)
# plot_multiple_models('DQN target networks - 300k steps', 'Option-critic', board_size='8 x 4', options=4)

# plot_lines_cleared('8 x 4', 'DQN target networks - 300k steps', 'tetris8x4-dqn-targetnets-400000steps')

# plot_any(['example results/scores/8 x 4/Attention OC/2 options/average.txt',
#           'example results/scores/8 x 4/Attention OC/4 options/average.txt',
#           'example results/scores/8 x 4/Option-critic/2 options/average.txt',
#           'example results/scores/8 x 4/Option-critic/4 options/average.txt',
#           'example results/scores/8 x 4/Option-critic/8 options/average.txt',
#           'example results/scores/8 x 4/PPOC/2 options/average.txt',
#           'example results/scores/8 x 4/PPOC/4 options/average.txt',
#           'example results/scores/8 x 4/PPOC/8 options/average.txt'],
#           'All OC models', max_eps=8000)

# plot_any(['example results/scores/20 x 10/Attention OC/2 options/average.txt',
#           'example results/scores/20 x 10/Attention OC/4 options/average.txt',
#           'example results/scores/20 x 10/Option-critic/2 options/average.txt',
#           'example results/scores/20 x 10/Option-critic/4 options/average.txt',
#           'example results/scores/20 x 10/Option-critic/8 options/average.txt',
#           'example results/scores/20 x 10/PPOC/2 options/average.txt',
#           'example results/scores/20 x 10/PPOC/4 options/average.txt',
#           'example results/scores/20 x 10/PPOC/8 options/average.txt'],
#           'All OC models', max_eps=2000)

# plot_multiple_runs('AOC-2 options-200000 steps', '20 x 10', 'Attention OC', 2, True)
# plot_multiple_runs('AOC-4 options-200000 steps', '20 x 10', 'Attention OC', 4, True)

# plot_multiple_models('PPO', 'PPOC', board_size='20 x 10', options='all')
# plot_multiple_models('PPO', 'PPOC', board_size='8 x 4', options='all', max_eps=8000)

#plot_multiple_models('Attention OC', 'PPO', 'DQN target networks - 200k steps', board_size='20 x 10', options=2, max_eps=1500)

# plot_other_envs(['CartPole - DQN-v1 - 100000 steps',
#     'CartPole - OC - 2 options - 100000 steps',
#     'CartPole - OC - 4 options - 100000 steps',
#     'CartPole - OC - 8 options - 100000 steps',
#     'CartPole - PPO-v0 - 100000 steps',
#     'CartPole - PPOC - 2 options - 100000 steps'],
#     'CartPole all agents', max_eps=1200)

# plot_other_envs(['FourRooms - DQN-v1 - 100000 steps',
#     'FourRooms - OC - 2 options - 100000 steps',
#     'FourRooms - OC - 4 options - 100000 steps',
#     'FourRooms - OC - 8 options - 100000 steps',
#     'FourRooms - PPO-v0 - 100000 steps',
#     'FourRooms - PPOC - 2 options - 100000 steps',
#     'FourRooms - PPOC - 4 options - 100000 steps',
#     'FourRooms - PPOC - 8 options - 100000 steps',],
#     'FourRooms all agents', max_eps=300)

# plot_other_envs(['FourRooms - DQN-v1 - 100000 steps',
#     'FourRooms - OC - 2 options - 100000 steps',
#     'FourRooms - OC - 4 options - 100000 steps',
#     'FourRooms - OC - 8 options - 100000 steps'],
#     'FourRooms DQN and OC')


# plot_other_envs(['CartPole - DQN-v1 - 100000 steps'], plot_name='Cartpole DQN')