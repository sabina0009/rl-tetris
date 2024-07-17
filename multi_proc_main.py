
import gym
import gym_simpletetris
import numpy as np
from ppo_torch import Agent
from utils import plot_learning_curve

env = gym.make('SimpleTetris-v0', reward_step=True)
N = 20
batch_size = 64
n_epochs = 4
alpha = 0.0003
agent = Agent(n_actions=env.action_space.n, batch_size=batch_size, 
                    alpha=alpha, n_epochs=n_epochs, 
                    input_dims=[env.observation_space.shape[0] * env.observation_space.shape[1]])

def run_worker(agent, process_num):
    n_games = 1000
    figure_file = 'plots/tetris1000games.png'
    best_score = env.reward_range[0]
    score_history = []
    learn_iters = 0
    avg_score = 0
    n_steps = 0
    np.random.seed(process_num)
    for i in range(n_games):
        observation = env.reset()
        observation = observation.flatten()
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            observation_ = observation_.flatten()
            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('process_num: ', process_num, 'episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                'time_steps', n_steps, 'learning_steps', learn_iters)
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)

    pass

threads = 8
import torch.multiprocessing as mproc
import threading
if __name__ == '__main__':
    #share the network weights between the processes
    agent.actor.share_memory()
    agent.critic.share_memory()
    processes = []
    UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
    ROLLING_EVENT.set()
    for process_num in range(threads):
            p = mproc.Process(target=run_worker, args=(agent, process_num))
            p.start()
            processes.append(p)
    for p in processes:
            p.join()