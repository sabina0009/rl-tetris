import argparse
from main_PPO import run_PPO
from main_PPO_sharednets import run_PPO_sharednets

parser = argparse.ArgumentParser(description="RL Tetris Pytorch")

parser.add_argument('--agent', default='PPO', help='Choose RL agent: PPO, DQN, OC, PPOC, AOC')
parser.add_argument('--boardsize', default = '8x4', help='20x10 or 8x4 Tetris board')
parser.add_argument('--steps', default=1000, help='Number of time steps to run agent for')
parser.add_argument('--runs', default=5, help='Number of runs of training')
parser.add_argument('--version', default=2, help='See versions for each agent in README.md')
parser.add_argument('--options', default=2, help = 'Number of options for OC agents')

if __name__ == '__main__':
    args = parser.parse_args()
    filename = f'Tetris {args.boardsize} - {args.agent}v{args.version} - {args.steps} steps'

    if args.agent == 'PPO':
        if args.version in [0, 1]:
            run_PPO(args, filename)
        if args.version == 2:
            run_PPO_sharednets(args, filename)