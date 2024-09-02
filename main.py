import argparse
from main_PPO import run_PPO
from main_PPO_sharednets import run_PPO_sharednets
from main_DQN import run_DQN
from main_DQN_targetnets import run_DQN_target
from main_OC import run_OC
from main_PPOC import run_PPOC

parser = argparse.ArgumentParser(description="RL Tetris Pytorch")

parser.add_argument('--agent', default='PPOC', help='Choose RL agent: PPO, DQN, OC, PPOC, AOC')
parser.add_argument('--boardsize', default = '20x10', help='20x10 or 8x4 Tetris board')
parser.add_argument('--steps', default=100, help='Number of time steps to run agent for')
parser.add_argument('--runs', default=5, help='Number of runs of training')
parser.add_argument('--version', default=0, help='See versions for each agent in README.md')
parser.add_argument('--options', default=2, help = 'Number of options for OC agents')

if __name__ == '__main__':
    args = parser.parse_args()
    filename = f'Tetris {args.boardsize} - {args.agent}-v{args.version} - {args.steps} steps'

    if args.agent == 'PPO':
        filename = f'Tetris {args.boardsize} - {args.agent}-v{args.version} - {args.steps} steps'
        if args.version in [0, 1]:
            run_PPO(args, filename)
        if args.version == 2:
            run_PPO_sharednets(args, filename)

    if args.agent == 'DQN':
        filename = f'Tetris {args.boardsize} - {args.agent}-v{args.version} - {args.steps} steps'
        if args.version == 0:
            run_DQN(args, filename)
        if args.version in [1, 2]:
            run_DQN_target(args, filename)

    if args.agent == 'OC':
        filename = f'Tetris {args.boardsize} - {args.agent} - {args.options} options - {args.steps} steps'
        run_OC(args, filename)

    if args.agent == 'PPOC':
        filename = f'Tetris {args.boardsize} - {args.agent} - {args.options} options - {args.steps} steps'
        run_PPOC(args, filename)

    if args.agent == 'AOC':
        filename = f'Tetris {args.boardsize} - {args.agent} - {args.options} options - {args.steps} steps'
