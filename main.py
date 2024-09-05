import argparse
from main_PPO import run_PPO
from main_PPO_sharednets import run_PPO_sharednets
from main_DQN import run_DQN
from main_DQN_targetnets import run_DQN_target
from main_OC import run_OC
from main_PPOC import run_PPOC
from main_AOC import run_AOC
from test_agents import test_agent

parser = argparse.ArgumentParser(description="RL Tetris Pytorch")

parser.add_argument('--environment', default='Tetris', help='Choose environment: Tetris, CartPole or FourRooms')
parser.add_argument('--agent', default='PPO', help='Choose RL agent: PPO, DQN, OC, PPOC, AOC')
parser.add_argument('--boardsize', default ='20x10', help='20x10 or 8x4 Tetris board')
parser.add_argument('--steps', default=100000, help='Number of time steps to run agent for')
parser.add_argument('--runs', default=5, help='Number of runs of training')
parser.add_argument('--version', default=0, help='See versions for each agent in README.md')
parser.add_argument('--options', default=1, help = 'Number of options for OC agents')
parser.add_argument('--testing', default=False, help='Train or test the agent.')
parser.add_argument('--testeps', default=10, help='How many episodes to test for.')

if __name__ == '__main__':
    args = parser.parse_args()

    filename = args.environment
    if filename == 'Tetris':
        filename += ' ' + args.boardsize
    if args.agent in ['PPO', 'DQN']:
        filename += f' - {args.agent}-v{args.version}'
    else:
        filename += f' - {args.agent} - {args.options} options'
    filename += f' - {args.steps} steps'
    
    if not args.testing:
        if args.environment == 'Tetris':
            if args.agent == 'PPO':
                if args.version in [0, 1]:
                    run_PPO(args, filename)
                if args.version == 2:
                    run_PPO_sharednets(args, filename)

            if args.agent == 'DQN':
                if args.version == 0:
                    run_DQN(args, filename)
                if args.version in [1, 2]:
                    run_DQN_target(args, filename)

            if args.agent == 'OC':
                run_OC(args, filename)

            if args.agent == 'PPOC':
                run_PPOC(args, filename)

            if args.agent == 'AOC':
                run_AOC(args, filename)
        
        if args.environment == 'CartPole' or 'FourRooms':
            if args.agent == 'PPO':
                if args.version in [0, 1]:
                    run_PPO(args, filename)
                if args.version == 2:
                    run_PPO_sharednets(args, filename)

            if args.agent == 'DQN':
                if args.version == 0:
                    run_DQN(args, filename)
                if args.version in [1, 2]:
                    run_DQN_target(args, filename)

            if args.agent == 'OC':
                run_OC(args, filename)

            if args.agent == 'PPOC':
                run_PPOC(args, filename)
        
    else:
        test_agent(args, filename)