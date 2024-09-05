from utils import make_env, to_tensor, get_column_heights, load_models
from ppo_torch import Agent as PPO_Agent
from dqn_targetnets_torch import Agent as DQN_Agent
from option_critic import OptionCriticFeatures as OC_Agent
from ppoc import OptionCriticFeatures as PPOC_Agent
from aoc import OptionCriticFeatures as AOC_Agent
import numpy as np


def test_agent(args, filename):
    environment = args.environment
    board_size = args.boardsize

    env, input_dims = make_env(environment, board_size)
    option_critic = args.agent in ['OC', 'PPOC', 'AOC']

    if  args.agent in ['PPO', 'DQN']:
        input_dims = [input_dims]

    if args.agent == 'PPO':
        agent = PPO_Agent(n_actions=env.action_space.n, input_dims=input_dims)
    #n_games = 1000
    elif args.agent == 'DQN':
        agent = DQN_Agent(n_actions=env.action_space.n, input_dims=input_dims)
    elif args.agent == 'OC':
        agent = OC_Agent(in_features=input_dims, num_actions=env.action_space.n, num_options=args.options)
    elif args.agent == 'PPOC':
        agent = PPOC_Agent(in_features=input_dims, num_actions=env.action_space.n, num_options=args.options)
    elif args.agent == 'AOC':
        agent = AOC_Agent(in_features=input_dims+args.options, num_actions=env.action_space.n, num_options=args.options)

    if args.agent in ['DQN', 'PPO']:
        agent.load_models(filename, process_num=0)
    elif args.agent in ['OC', 'AOC']:
        load_models(agent, process_num=0, filename=filename)
    else:
        agent.load_checkpoint(f'{filename} 0')

    steps = 0
    max_eps = args.testeps
    episode = 0

    while episode < max_eps:
        rewards = 0
        done = False
        current_option = 0
        option_termination = True

        if args.environment == 'FourRooms':
            obs = env.reset()
        else:
            obs, info   = env.reset()
        if args.environment == 'Tetris':
            obs = obs.flatten()

        env.render()

        if args.agent == 'AOC':
            additional_feature_len = agent.num_options
            num_holes = env.engine.holes 
            piece_height = sum(np.any(env.engine.board, axis=0)) 
            heights = get_column_heights(env.engine.board) 
            aggregate_heights = sum(np.any(env.engine.board, axis=0)) 
            lines = env.engine.lines_cleared
            bumpiness = sum([abs(heights[i]-heights[i+1]) for i in range(len(heights)-1)])
            if additional_feature_len == 2:
                additional_features = np.array([num_holes, piece_height]) 
            if additional_feature_len == 4:
                additional_features = np.array([num_holes, aggregate_heights, lines, bumpiness]) 
            obs = np.concatenate([obs.flatten(), additional_features]).flatten() 

        if args.agent in ['OC', 'AOC']:
            state = agent.get_state(to_tensor(obs))
            greedy_option  = agent.greedy_option(state)

        while not done:
            ep_len = 0

            if args.agent in ['OC', 'AOC']:
                epsilon = agent.epsilon

            if option_termination and args.agent == 'PPOC':
                current_option = agent.get_next_option(state)
            elif option_termination and args.agent in ['OC', 'AOC']:
                current_option = np.random.choice(agent.num_options) if np.random.rand() < epsilon else greedy_option

            if args.agent =='PPO':
                action, _, _ = agent.choose_action(obs)
            elif args.agent == 'DQN':
                action = agent.choose_action(obs)
            elif args.agent == 'OC':
                action, _, _ = agent.choose_action(state, current_option)
            else:
                action, _, _ = agent.choose_action(obs, current_option)

            if args.environment == 'FourRooms':
                next_obs, reward, done, _ = env.step(action)
            else:
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            if args.environment == 'Tetris':
                next_obs = next_obs.flatten()

            env.render()

            obs = next_obs
            steps += 1
            ep_len += 1
            rewards += reward

            if args.agent == 'AOC':
                num_holes = env.engine.holes 
                piece_height = sum(np.any(env.engine.board, axis=0)) 
                heights = get_column_heights(env.engine.board) 
                aggregate_heights = sum(np.any(env.engine.board, axis=0)) 
                lines = env.engine.lines_cleared
                bumpiness = sum([abs(heights[i]-heights[i+1]) for i in range(len(heights)-1)])
                if additional_feature_len == 2:
                    additional_features = np.array([num_holes, piece_height]) 
                if additional_feature_len == 4:
                    additional_features = np.array([num_holes, aggregate_heights, lines, bumpiness]) 
                obs = np.concatenate([obs.flatten(), additional_features]).flatten() 

        if args.environment == 'FourRooms':
            rewards = ep_len
        if args.environment == 'Tetris':
            lines_cleared = info['lines_cleared']
            print(f'Episode {episode} | Score {rewards} | Lines cleared {lines_cleared}')
        else:
            print(f'Episode {episode} | Score {rewards}')
        episode += 1

            

