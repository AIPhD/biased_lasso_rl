import numpy as np
import matplotlib.pyplot as plt
import config as c


def plot_cumulative_rewards_per_segment(total_reward,
                                        episode,
                                        seg_len,
                                        no_segment,
                                        x_label="episodes",
                                        y_label="Accumulated reward per episode",
                                        directory=None,
                                        plot_suffix='test_run'):
    '''Plot total reward per episode as a function of episodes.'''

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    for i in range (no_segment - 1):
        plt.axvline((i + 1)*seg_len, linestyle="--", color="r")

    plt.plot(np.arange(episode) + 1,
             total_reward)
    plt.show()

    if directory is not None:
        plt.savefig(f'{c.PLOT_DIR}{directory}/{plot_suffix}.pdf',
                    dpi=800,
                    format= "pdf",
                    bbox_inches='tight',
                    pad_inches=0)

    plt.close()

def plot_time_steps_in_maze(total_time_steps,
                            episode,
                            seg_len,
                            no_segments,
                            x_label='Episode',
                            y_label='Time steps required',
                            directory=None,
                            plot_suffix='steps_required_per_episode'):
    '''Plots total time steps required for agent to reach the end of the labyrinth'''

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    for i in range (no_segments -1):
        plt.axvline((i+1)*seg_len, linestyle='--', color='r')

    plt.plot(np.arange(episode) + 1,
             total_time_steps)
    plt.show()

    if directory is not None:
        plt.savefig(f'{c.PLOT_DIR}{directory}/{plot_suffix}.pdf')

    plt.close()
