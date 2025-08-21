import numpy as np
import matplotlib.pyplot as plt
import config as c


def plot_cumulative_rewards_per_segment(total_reward,
                                        episode,
                                        seg_len,
                                        no_segment,
                                        x_label="episodes",
                                        y_label="Accumulated reward per episode",
                                        plot_suffix='accumulated_reward',
                                        plot_dir=None):
    '''Plot total reward per episode as a function of episodes.'''

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    for i in range (no_segment - 1):
        plt.axvline((i + 1)*seg_len, linestyle="--", color="r")

    plt.plot(np.arange(episode) + 1,
             total_reward)
    # plt.show()

    if plot_dir is not None:
        plt.savefig(f'{plot_dir}/{plot_suffix}.pdf',
                    dpi=800,
                    format= "pdf",
                    bbox_inches='tight',
                    pad_inches=0)

    plt.close()


def plot_moving_average_reward(total_reward,
                               seg_len,
                               no_segment,
                               plot_label,
                               mean_length=25):
    '''Plot the moving average of total rewards per episode as a function of episodes.'''


    for i in range (no_segment - 1):
        plt.axvline((i + 1)*seg_len - mean_length, linestyle="--", color="r")

    running_average_reward = moving_average(total_reward, mean_length)

    plt.plot(np.arange(len(running_average_reward)) + mean_length,
             running_average_reward,
             label=plot_label)


def plot_multiple_moving_average_rewards(reward_lists,
                                         legends,
                                         x_label="episodes",
                                         y_label="Accumulated mean reward per episode",
                                         plot_suffix='accumulated_mean_reward',
                                         plot_dir=None,
                                         mean_length=25):
    '''Plot multiple moving average rewards in one plot.'''

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    x_range = np.min([len(rewards) for rewards in reward_lists])

    for rewards, legend in zip(reward_lists, legends):
        plot_moving_average_reward(rewards[:x_range],
                                   x_range,
                                   no_segment=1,
                                   plot_label=legend,
                                   mean_length=mean_length)
    
    plt.legend(legends)
    plt.tight_layout()
        
    if plot_dir is not None:
        plt.savefig(f'{plot_dir}/{plot_suffix}.pdf',
                    dpi=800,
                    format= "pdf",
                    bbox_inches='tight',
                    pad_inches=0)
    
    plt.close()


def moving_average(inp_array, length):
    '''Calculate moving average given an array and length over which to calculate the mean.'''

    cumsum = np.cumsum(np.insert(inp_array, 0, 0))
    return (cumsum[length:]-cumsum[:-length])/length


def plot_time_steps_in_maze(total_time_steps,
                            episode,
                            seg_len,
                            no_segments,
                            x_label='Episode',
                            y_label='Time steps required',
                            plot_suffix='steps_required_per_episode',
                            plot_dir=None):
    '''Plots total time steps required for agent to reach the end of the labyrinth'''

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    for i in range (no_segments -1):
        plt.axvline((i+1)*seg_len, linestyle='--', color='r')

    plt.plot(np.arange(episode) + 1,
             total_time_steps)
    # plt.show()

    if plot_dir is not None:
        plt.savefig(f'{plot_dir}/{plot_suffix}.pdf')

    plt.close()


def plot_loss_function(loss_values,
                       time_step,
                       x_label='Time step',
                       y_label='Loss',
                       plot_suffix='Loss_evolution',
                       plot_dir=None):
    '''Plots loss as a funtion of time steps.'''

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.plot(np.arange(time_step) + 1,
             loss_values),

    if plot_dir is not None:
        plt.savefig(f'{plot_dir}/{plot_suffix}.pdf')

    plt.close()

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current Axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def plot_nn_weights(fc_network, seg_ind, dir):
    '''Plot the weights of given neural network in a heatmap.'''

    for key in fc_network.state_dict():

        weights = fc_network.state_dict()[key].detach().cpu().numpy()

        if isinstance(weights[0], np.float32):
            weights = weights[:, np.newaxis]

        plt.imshow(weights, cmap='seismic', vmin=-np.max(np.abs(weights)), vmax=np.max(np.abs(weights)))
        plt.colorbar()
        plt.savefig(f'{dir}/{key}_{seg_ind}.pdf', dpi=300)
        plt.close()
    