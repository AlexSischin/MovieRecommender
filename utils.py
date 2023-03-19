import matplotlib.pyplot as plt

colors = ['#fd7f6f', '#7eb0d5', '#b2e061', '#bd7ebe', '#ffb55a', '#ffee65', '#beb9db', '#fdcce5', '#8bd3c7']
l_styles = ['solid', 'dotted', 'dashed', 'dashdot']


def get_single_axes() -> tuple[plt.Figure, plt.Axes]:
    return plt.subplots()


def compare_history(history_list, metrics):
    fig, ax = get_single_axes()
    ax.set_title('Learning curve')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(' / '.join(metrics))

    for history, color in zip(history_list, colors):
        model = history.model.name
        epoch = history.epoch
        history_metrics = history.history
        for metric, style in zip(metrics, l_styles):
            metric_hist = history_metrics[metric]
            label = f'{model} - {metric}'
            ax.plot(epoch, metric_hist, label=label, color=color, linestyle=style)
    fig.legend()
    plt.show()
