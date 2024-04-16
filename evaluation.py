import numpy as np
import matplotlib.pyplot as plt
import config as c


def plot_cumulative_rewards(total_reward,
                            epochs=c.EPOCHS,
                            x_label="epochs",
                            y_label="Accumulated reward per epoch",
                            save_plot=c.SAVE_PLOT,
                            directory='',
                            plot_suffix='test_run'):

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(np.arange(epochs) + 1,
             total_reward)
    plt.show()

    if save_plot:
        plt.savefig(f'{c.PLOT_DIR}{directory}/{plot_suffix}.pdf',
                    dpi=800,
                    format= "pdf",
                    bbox_inches='tight',
                    pad_inches=0)

    plt.close()
    return
