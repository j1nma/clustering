## Clustering
This work experiments with three clustering techniques in Python, including one from either hierarchical and partitional
techniques.

### Installation
```shell
$ pip3 install -r requirements.txt
```

### Running

Custom hyperparameters in a textfile i.e. _"./configs/config.txt"_.

```shell
$ python3 experiments.py ./configs/config.txt
```

A _results_ folder will contain a timestamp directory with the latest results.

### Datasets
* Iris (http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html) 
* Breast Cancer (http://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic))
* Toy Dataset: Noisy circles (https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html)

### Techniques
* Hierarchical: Agglomerative Clustering
* Partitional: K-Means
* Partitional: Gaussian Mixture

### Report
Clustering-Alonso.pdf