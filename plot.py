from maze_project import plot_maze as pm
from atari_project import plot_atari as pa



def main():
    '''Script to plot all accumulated rewards and losses of different tasks.'''

    pm.plot_baselines()
    # pa.plot_baselines()

if __name__ == '__main__':
    main()