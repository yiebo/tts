import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# plt.switch_backend('agg')


def plot_heatmap(input, x_label=None, y_label=None):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.imshow(input.numpy(), interpolation='none')
    fig.colorbar(cax)

    # Set up axes
    if x_label is not None:
        ax.set_xticklabels(x_label, rotation=90)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    if y_label is not None:
        ax.set_yticklabels(y_label)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    return fig
