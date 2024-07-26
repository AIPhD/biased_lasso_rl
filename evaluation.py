import numpy as np
import matplotlib.pyplot as plt
import config as c


def plot_cumulative_rewards_per_segment(total_reward,
                                        epochs,
                                        seg_len,
                                        no_segment,
                                        x_label="epochs",
                                        y_label="Accumulated reward per epoch",
                                        directory=None,
                                        plot_suffix='test_run'):
    '''Plot total reward per epochs as a function of epochs.'''

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    for i in range (no_segment - 1):
        plt.axvline((i + 1)*seg_len, linestyle="--", color="r")

    plt.plot(np.arange(epochs) + 1,
             total_reward)
    plt.show()

    if directory is not None:
        plt.savefig(f'{c.PLOT_DIR}{directory}/{plot_suffix}.pdf',
                    dpi=800,
                    format= "pdf",
                    bbox_inches='tight',
                    pad_inches=0)

    plt.close()
