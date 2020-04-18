import random
import seaborn as sns
import matplotlib._color_data as mcd
import numpy as np
from matplotlib import pyplot as plt

overlap = tuple(name for name in mcd.CSS4_COLORS
                if "xkcd:" + name in mcd.XKCD_COLORS)

FIGURES_DIR = 'figures/'


def gen_colors(cluster_len):
    colors = []
    while len(colors) != cluster_len:
        item = random.choice(overlap)
        if item not in colors:
            colors.append(item)
    return colors


def plot_data(df, colors):
    labels = list(df['labels'].unique())

    for label in labels:
        x_y = df.loc[df['labels'] == label, ['x', 'y']]
        plt.plot(list(x_y['x']), list(x_y['y']), 'ro', color=colors[label])

    for i, row in df.iterrows():
        plt.annotate(str(int(row['labels'])), (row['x'], row['y']))

    plt.axis([-0.5, 18, -0.5, 18])
    plt.title('Fake Generated Clusters for KNeighborsClassifier model')
    plt.savefig(FIGURES_DIR + 'Figure_data' + '.png')
    plt.show()


def plot_prediction(model, df, point: tuple, colors):
    labels = list(df['labels'].unique())

    for label in labels:
        x_y = df.loc[df['labels'] == label, ['x', 'y']]
        plt.plot(list(x_y['x']), list(x_y['y']), 'ro', color=colors[label])

    for i, row in df.iterrows():
        plt.annotate(str(int(row['labels'])), (row['x'], row['y']))

    plt.axis([-0.5, 18, -0.5, 18])
    x, y = point[0], point[1]

    plt.plot(x, y, 'ro', marker='*', markersize=20)

    pred = model.predict(np.array([[x, y]]))[0]

    plt.title(f'predict for {x},{y}   class is {pred}')

    plt.savefig(FIGURES_DIR + f'Figure_pred_{x}_{y}' + '.png')
    plt.show()


def plot_cm(cm):
    sns.heatmap(cm / np.sum(cm), annot=True, fmt='.2%', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig(FIGURES_DIR+'Figure_cm'+'.png')
    plt.show()


def plot_optimal_k(k_range, scores, opt_k, opt_score):
    fig, ax = plt.subplots()
    ax.plot(k_range, scores)
    ax.plot(opt_k, opt_score, 'ro', color='green', markersize=10)
    ax.annotate('   optimal k', (opt_k, opt_score))
    ax.set(xlabel='Possible K neighbour', ylabel='Accuracy',
           title=f'Optimal K: {opt_k} Optimal Score: {opt_score}')
    plt.savefig(FIGURES_DIR + 'Figure_opt_k'+'.png')
    plt.show()
