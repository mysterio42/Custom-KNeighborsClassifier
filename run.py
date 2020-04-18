import numpy as np
import pandas as pd
import argparse
from utils.data import CustomData
from utils.model import train_model, load_model,find_optimal_k
from utils.plot import plot_data, plot_prediction,plot_optimal_k


def parse_args():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected')

    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str2bool, default=True,
                        help='True: Load trained model  False: Train model default: True')
    parser.add_argument('--gen', type=str2bool, default=True,
                        help='True: Generate Fake data False: Use already generated data')
    parser.print_help()
    return parser.parse_args()


if __name__ == '__main__':

    np.random.seed(0)

    args = parse_args()

    if args.gen:
        data = CustomData(path='data/',nums=100)
        data.generate_cluster(x_range=(1, 3), y_range=(4, 8), cluster_key=0)
        data.generate_cluster(x_range=(3, 6), y_range=(2, 7), cluster_key=1)
        data.generate_cluster(x_range=(6, 8), y_range=(5, 8), cluster_key=2)
        data.generate_cluster(x_range=(3, 6), y_range=(8, 11), cluster_key=3)
        data.generate_cluster(x_range=(7, 9), y_range=(2, 5), cluster_key=4)
        data.generate_cluster(x_range=(8, 10), y_range=(9, 12), cluster_key=5)
        data.generate_cluster(x_range=(9, 12), y_range=(4, 8), cluster_key=5)
        data.generate_cluster(x_range=(2, 6), y_range=(11, 14), cluster_key=6)
        data.generate_cluster(x_range=(6, 9), y_range=(8, 12), cluster_key=7)
        data.generate_cluster(x_range=(10, 12), y_range=(8, 11), cluster_key=8)
        data.generate_cluster(x_range=(9, 12), y_range=(11, 13), cluster_key=9)
        data.to_csv(name='clusters')

    df = pd.read_csv('data/clusters.csv')

    colors = CustomData.generate_colors(df)

    if args.load:
        model = load_model()
        plot_data(df,colors)
        plot_prediction(model, df, (2.3, 9.1),colors)
        plot_prediction(model, df, (9.7, 8.1),colors)
        plot_prediction(model, df, (9.3, 10.2),colors)
    else:
        plot_data(df,colors)

        features, labels = CustomData.load_data(df)

        k_range, scores, opt_k, opt_score = find_optimal_k(features, labels)
        plot_optimal_k(k_range, scores, opt_k, opt_score)

        model = train_model(features, labels, opt_k)


        plot_prediction(model, df, (2.3, 9.1),colors)
        plot_prediction(model, df, (9.7, 8.1),colors)
        plot_prediction(model, df, (9.3, 10.2),colors)