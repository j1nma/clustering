import argparse
import datetime
import warnings

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
import time
from itertools import cycle, islice
from sklearn import cluster
from sklearn.datasets import load_iris, load_breast_cancer, make_circles
from sklearn.manifold import TSNE, MDS
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path


def log(logfile, s):
    """ Log a string into a file and print it. """
    with open(logfile, 'a', encoding='utf8') as f:
        f.write(str(s))
        f.write("\n")
    print(s)


def get_args_parser():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument(
        "-d",
        "--dataset",
        default="Iris",
        help="Name of the dataset to use: Iris, BreastCancer, NoisyCircles."
    )
    parser.add_argument(
        "-t",
        "--technique",
        default="Agglomerative",
        help="Name of the clustering technique: Agglomerative, kMeans, GaussianMixture."
    )
    parser.add_argument(
        "-k",
        "--clusters",
        default=2,
        help="Number of clusters for the given technique."
    )
    parser.add_argument(
        "-s",
        "--seed",
        default=1910299034,
        help="Random seed."
    )
    parser.add_argument(
        "-nn",
        "--kneighbours",
        default=10,
        help="Number of neighbors for each sample of the kNN graph needed for Agglomerative technique"
    )
    parser.add_argument(
        "-od",
        "--outdir",
        default='results/'
    )

    return parser


def experiments(config_file):
    args = get_args_parser().parse_args(['@' + config_file])

    # Set seed
    np.random.seed(int(args.seed))

    # Construct output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = args.outdir + timestamp + '/'

    # Create results directory
    outdir_path = Path(outdir)
    if not outdir_path.is_dir():
        os.makedirs(outdir)

    # Logging
    logfile = outdir + 'log.txt'

    log(logfile, "Directory " + outdir + " created.")

    # Set dataset
    if args.dataset == 'Iris':
        dataset = load_iris()
        dataset_name = "Iris"
        original_labels = ['setosa', 'versicolour', 'virginica']
    elif args.dataset == 'BreastCancer':
        dataset = load_breast_cancer()
        dataset_name = "Breast Cancer Wisconsin"
        original_labels = ['malignant', 'benign']
    elif args.dataset == 'NoisyCircles':
        n_samples = 1500
        noisy_circles = make_circles(n_samples=n_samples, factor=.5, noise=.05)
        dataset = noisy_circles
        dataset_name = "NoisyCircles"
    else:
        raise ("Dataset not found")

    if dataset_name != "NoisyCircles":
        data = dataset.data
        labels = dataset.target

    # Add labels and feature names
    # df = pd.DataFrame(data, columns=dataset.feature_names)
    # df['label'] = labels
    # df['label'] = df['label'].apply(lambda i: str(i))
    # for i in range(0, len(original_labels)):
    #     df['label'].replace(str(i), original_labels[i], inplace=True)

    # Dataset analysis
    # log(logfile, 'Size of the data: {} and labels: {}'.format(data.shape, labels.shape))
    # log(logfile, 'Size of the reshaped dataframe: {}'.format(df.shape))
    # log(logfile, df.head())
    # log(logfile, df.tail())

    # Normalization of features
    # data = df.loc[:, dataset.feature_names].values
    # data = StandardScaler().fit_transform(data)

    # Set number of components
    # n_components = int(args.components)

    X, y = dataset

    plot_num = 1

    # Set technique
    if args.technique == 'Agglomerative':

        connectivity = kneighbors_graph(X, n_neighbors=int(args.kneighbours), include_self=False)

        average_linkage = cluster.AgglomerativeClustering(
            linkage="average",
            affinity="cityblock",
            n_clusters=2,
            connectivity=connectivity)

        algorithm = average_linkage

        time_start = time.time()

        # catch warnings related to kneighbors_graph
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="the number of connected components of the " +
                        "connectivity matrix is [0-9]{1,2}" +
                        " > 1. Completing it to avoid stopping the tree early.",
                category=UserWarning)
            warnings.filterwarnings(
                "ignore",
                message="Graph is not fully connected, spectral embedding" +
                        " may not work as expected.",
                category=UserWarning)
            # average_linkage.fit(X)
            algorithm.fit(X)

        # log(logfile, 'Cumulative explained variation for {} principal components: {}'.format(n_components,
        #                                                                                      np.sum(
        #                                                                                          pca.explained_variance_ratio_).round(
        #                                                                                          decimals=3)))

        time_stop = time.time()
        if hasattr(algorithm, 'labels_'):
            y_pred = algorithm.labels_.astype(np.int)
        else:
            y_pred = algorithm.predict(X)

        # plt.subplot(len(datasets), len(clustering_algorithms), plot_num)
        plt.subplot(3, 3, plot_num)
        # if i_dataset == 0:
        if plot_num == 1:
            plt.title("{} of {} Dataset".format(args.technique, dataset_name), size=18)

        colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(y_pred) + 1))))
        # add black color for outliers (if any)
        colors = np.append(colors, ["#000000"])
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.xticks(())
        plt.yticks(())
        plt.text(.99, .01, ('%.2fs' % (time_stop - time_start)).lstrip('0'),
                 transform=plt.gca().transAxes, size=15,
                 horizontalalignment='right')
        plot_num += 1


    elif args.technique == 'kMeans':
        time_start = time.time()
        tsne = TSNE(n_components=n_components, n_iter=1000, random_state=int(args.seed))
        data_transformed = tsne.fit_transform(data)
        log(logfile, 't-SNE done! Time elapsed: {0:.3f} seconds'.format(time.time() - time_start))
        transformed_df = pd.DataFrame(data=data_transformed, columns=['PC ' + str(i + 1) for i in range(n_components)])

    elif args.technique == 'GaussianMixture':
        embedding = MDS(n_components=n_components)
        data_transformed = embedding.fit_transform(data)
        log(logfile, 'MDS transformation shape: {}'.format(data_transformed.shape))
        transformed_df = pd.DataFrame(data=data_transformed, columns=['PC ' + str(i + 1) for i in range(n_components)])

    else:
        raise ("Technique not found")

    # log(logfile, transformed_df.tail())

    # Plotting
    # if n_components == 3:
    #     fig = plt.figure(figsize=(10, 10))
    #     ax = Axes3D(fig)
    #     ax.set_zlabel('PC 3', fontsize=15)
    # elif n_components == 2:
    #     plt.figure(figsize=(10, 10))
    #
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=14)
    # plt.xlabel('PC 1', fontsize=15)
    # plt.ylabel('PC 2', fontsize=15)
    #
    # plt.title("{} of {} Dataset".format(args.technique, dataset_name), fontsize=20)
    # colors = ['r', 'g', 'b']
    # for label, color in zip(original_labels, colors):
    #     indicesToKeep = df['label'] == label
    #     if n_components == 3:
    #         ax.scatter(transformed_df.loc[indicesToKeep, 'PC 1'],
    #                    transformed_df.loc[indicesToKeep, 'PC 2'],
    #                    transformed_df.loc[indicesToKeep, 'PC 3'], c=color, s=50)
    #     else:
    #         plt.scatter(transformed_df.loc[indicesToKeep, 'PC 1'],
    #                     transformed_df.loc[indicesToKeep, 'PC 2'], c=color, s=50)

    # if args.dataset == 'Iris':
    #     plt.legend(original_labels, prop={'size': 15}, loc="lower right")
    # else:
    #     plt.legend(original_labels, prop={'size': 15}, loc="upper right")

    plt.savefig(outdir + '{}_k={}.svg'.format(args.technique, args.clusters), format="svg")


if __name__ == "__main__":
    experiments(config_file=sys.argv[1])
