import matplotlib.pyplot as plt
import numpy as np

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


def compare_distributions(true, predictions, bins=10, rwidth=0.7):
    fig, ax = get_single_axes()
    ax.set_title('Output distribution')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')

    if predictions:
        labels, data, cols = list(zip(*[[a[0], np.array(a[1]).flatten(), c] for a, c in zip(predictions, colors)]))
        labels, data, cols = list(labels), list(data), list(cols)
    else:
        labels, data, cols = [], [], []

    labels.insert(0, 'True')
    data.insert(0, np.array(true).flatten())
    cols.insert(0, '#333333')

    ax.hist(data, bins=bins, density=True, color=cols, alpha=1, rwidth=rwidth, label=labels)
    fig.legend()
    plt.show()


def get_cossim_matrix(in_arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(in_arr, axis=1)
    in_arr_norm = in_arr / norms[..., np.newaxis]
    similarity_map = in_arr_norm @ in_arr_norm.T
    masked_sim_map = np.triu(similarity_map, 1)
    return masked_sim_map


def argsort2d(matrix: np.ndarray) -> np.ndarray:
    sorted_flat_ids = np.unravel_index(np.argsort(-matrix, axis=None), matrix.shape)
    sorted_2d_ids = np.stack(sorted_flat_ids, axis=0).T
    return sorted_2d_ids
