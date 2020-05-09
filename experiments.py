import argparse
import datetime
import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
from itertools import cycle, islice
from scipy.cluster.hierarchy import dendrogram
from sklearn import cluster, mixture, metrics
from sklearn.datasets import load_iris, load_breast_cancer, make_circles
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
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

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

# Returns indices where labels taken a particular cluster number
def get_cluster_indices(cluster_number, labels):
    return np.where(cluster_number == labels)[0]

# Computes entropy for a specific cluster TODO cite
def cluster_entropy(cluster_types):
    length = len(cluster_types)
    probs = {elem:float(sum([elem == s for s in cluster_types]))/length for elem in set(cluster_types)}
    h = -sum([probs[p] * math.log(probs[p]) for p in probs])
    return h

# Calculates the entropy of each cluster, and returns a list of group entropies TODO cite
def all_cluster_entropy(all_cluster_types, data_size):
    return sum([cluster_entropy(cluster_types) * (cluster_types.size/data_size) for cluster_types in all_cluster_types])

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
    n_samples = 1500
    noisy_circles = make_circles(n_samples=n_samples, factor=.5, noise=.05)

    plt.figure(figsize=(7 * 2 + 3, 12.5))
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05, hspace=.01)
    plot_num = 1

    datasets = (
        (3, load_iris(return_X_y=True)),
        # (2, load_breast_cancer(return_X_y=True)),
        # (2, noisy_circles)
    )

    # TODO: add comments and cite original script from sklearn
    for i_dataset, (n_clusters, dataset) in enumerate(datasets):
        X, y = dataset

        # Normalization of features for easier parameter selection
        X = StandardScaler().fit_transform(X)

        connectivity = kneighbors_graph(X, n_neighbors=int(args.kneighbours), include_self=False)
        connectivity = 0.5 * (connectivity + connectivity.T) # Make connectivity symmetric

        average_linkage = cluster.AgglomerativeClustering(
            linkage="average",
            affinity="cityblock",
            n_clusters=n_clusters,
            connectivity=connectivity)

        ward_linkage = cluster.AgglomerativeClustering(
            linkage="ward",
            n_clusters=n_clusters)

        complete_linkage = cluster.AgglomerativeClustering(
            linkage="complete",
            n_clusters=n_clusters)

        single_linkage = cluster.AgglomerativeClustering(
            linkage="single",
            n_clusters=n_clusters)

        k_means = cluster.KMeans(n_clusters=n_clusters)

        gaussian_mixture = mixture.GaussianMixture(
            n_components=n_clusters,
            covariance_type='full')

        # Set techniques
        techniques = (
            ('Agglomerative Avg', average_linkage),
            # ('Agglomerative Single', single_linkage),
            # ('Agglomerative Complete', complete_linkage),
            # ('Agglomerative Ward', ward_linkage),
            # ('kMeans', k_means),
            # ('GaussianMixture', gaussian_mixture),
        )

        for name, technique in techniques:
            time_start = time.time()

            # Catch warnings related to kneighbors_graph
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
                technique.fit(X)

            time_stop = time.time()
            if hasattr(technique, 'labels_'):
                y_pred = technique.labels_.astype(np.int)
            else:
                y_pred = technique.predict(X)

            plt.subplot(len(datasets), len(techniques), plot_num)
            if i_dataset == 0:
                plt.title("{}".format(name), size=15)

            colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a']),int(max(y_pred) + 1))))
            colors = np.append(colors, ["#000000"]) # Add black color for outliers (if any)
            plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred], alpha=0.60)

            plt.xlim(-2.5, 2.5)
            plt.ylim(-2.5, 2.5)
            plt.xticks(())
            plt.yticks(())
            plt.text(.99, .01, ('%.2fs' % (time_stop - time_start)).lstrip('0'),
                     transform=plt.gca().transAxes, size=15,
                     horizontalalignment='right')
            plot_num += 1

            # Metrics
            # Entropy
            all_cluster_types = [y[get_cluster_indices(c, y_pred)] for c in range(n_clusters)]
            all_entropies = all_cluster_entropy(all_cluster_types, y.shape[0])

            # F-Score
            fscore = metrics.f1_score(y, y_pred, average='micro')
            a = 0

    # Plotting
    plt.savefig(outdir + 'plot.svg', format="svg")


def plot_agglomerative_dendograms(config_file):
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

    # Set dataset
    n_samples = 1500
    noisy_circles = make_circles(n_samples=n_samples, factor=.5, noise=.05)

    plt.figure(figsize=(7 * 2 + 3, 12.5))
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05, hspace=.01)
    plot_num = 1

    datasets = (
        (3, load_iris(return_X_y=True)),
        (2, load_breast_cancer(return_X_y=True)),
        (2, noisy_circles)
    )

    for i_dataset, (n_clusters, dataset) in enumerate(datasets):
        X, y = dataset

        # Normalization of features for easier parameter selection
        X = StandardScaler().fit_transform(X)

        connectivity = kneighbors_graph(X, n_neighbors=int(args.kneighbours), include_self=False)
        connectivity = 0.5 * (connectivity + connectivity.T) # Make connectivity symmetric

        # setting distance_threshold=0 ensures we compute the full tree. TODO cite source
        average_linkage = cluster.AgglomerativeClustering(
            linkage="average",
            affinity="cityblock",
            distance_threshold=0, n_clusters=None,
            connectivity=connectivity)

        ward_linkage = cluster.AgglomerativeClustering(
            linkage="ward",
            distance_threshold=0, n_clusters=None,)

        complete_linkage = cluster.AgglomerativeClustering(
            linkage="complete",
            distance_threshold=0, n_clusters=None,)

        single_linkage = cluster.AgglomerativeClustering(
            linkage="single",
            distance_threshold=0, n_clusters=None,)

        techniques = (
            ('Agglomerative Avg', average_linkage),
            ('Agglomerative Single', single_linkage),
            ('Agglomerative Complete', complete_linkage),
            ('Agglomerative Ward', ward_linkage)
        )

        for name, technique in techniques:
            model = technique.fit(X)

            plt.subplot(len(datasets), len(techniques), plot_num)
            if i_dataset == 0:
                plt.title("{}".format(name), size=15)

            plot_dendrogram(model, truncate_mode='level', p=n_clusters)
            # plt.xlabel("Number of points in node (or index of point if no parenthesis).") TODO: explain
            plot_num += 1

    # Plotting
    plt.savefig(outdir + 'agglomerative_dendrograms.svg', format="svg")

if __name__ == "__main__":
    experiments(config_file=sys.argv[1])
    # plot_agglomerative_dendograms(config_file=sys.argv[1])
