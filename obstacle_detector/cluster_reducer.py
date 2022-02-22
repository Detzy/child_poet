from __future__ import print_function
import os
import numpy as np
import scipy.io as sio
import collections
import pandas as pd

try:
    import cPickle as pickle
except ImportError:
    import pickle


def threshold_clusters(clustering, file_names, threshold):
    """
    Method to split clustering into those above and below some threshold.
    Only clusters larger than (or equal to) the threshold is allowed on the "above" list.

    Parameters
    ----------
    clustering  :   iterable object
                    1D list/numpy array with assigned cluster labels
    file_names  :   iterable object
                    1D list/numpy array with the filenames of the images associated with each label in "clustering"
    threshold   :   int
                    Threshold of size of clusters. Determines what clusters will be in each of the returned lists.

    Returns
    -------
    above   :   list
                2D lists with labels and associated filenames, for clusters above/equal to threshold
    below   :   list
                2D lists with labels and associated filenames, for clusters below threshold
    cluster_sizes   :   Counter-object
                        Contains the sizes of each type of cluster
    """

    cluster_sizes = collections.Counter(clustering)

    z_files = zip(clustering, file_names)
    above = [[cluster, filename.strip()] for cluster, filename in z_files if cluster_sizes[cluster] >= threshold]

    z_files = zip(clustering, file_names)
    below = [[cluster, filename.strip()] for cluster, filename in z_files if cluster_sizes[cluster] < threshold]

    return above, below, cluster_sizes


def make_cluster_values_adjacent(in_clustering):
    """
    Reduces a cluster distribution to one where every cluster number is adjacent to another, starting at 0.
    This means that if there are n=10 clusters, every cluster will be guaranteed to be the values 0 to 9.

    Parameters
    ----------
    in_clustering  :    list
                        1D list with assigned cluster labels

    Returns
    -------
    out_clustering  :   list
                        1D list of same length as input_clustering, but with adjacent cluster values
    """
    out_clustering = in_clustering.copy()

    next_cluster = 0
    for i in range(max(in_clustering) + 1):
        if i in in_clustering:
            out_clustering = [next_cluster if c == i else c for c in out_clustering]
            next_cluster += 1

    return out_clustering


def scale_cluster_zero(filenames, clustering, size_multiplier=4):
    """
    Scales cluster zero to a multiple of the second largest class

    Parameters
    ----------
    filenames           :   list of str
                            list of strings containing file names of images
    clustering          :   list of int
                            list of integers giving the cluster assignment of each image in filenames
    size_multiplier     :   int, default=4
                            Determines how to scale cluster 0. Cluster 0 will have, at maximum, as many images as
                            size_multiplier times the size of the second largest cluster.

    Returns
    -------
    scaled_filenames    : list of str
    scaled_clustering   : list of str
    """
    scaled_filenames = []
    scaled_clustering = []

    cluster_sizes = collections.Counter(clustering)
    size_of_class_0_prior = cluster_sizes.most_common(2)[0][1]
    second_largest_cluster_size = cluster_sizes.most_common(2)[1][1]
    size_of_class_zero = size_multiplier * second_largest_cluster_size

    print("Reducing size of class 0 from {} to {} based on second largest cluster (size {})".format(
        size_of_class_0_prior, size_of_class_zero, second_largest_cluster_size
    ))

    admitted = 0
    for fn, cluster in zip(filenames, clustering):
        if cluster == 0:
            if admitted >= size_of_class_zero:
                continue
            else:
                admitted += 1
                scaled_filenames.append(fn)
                scaled_clustering.append(cluster)
        else:
            scaled_filenames.append(fn)
            scaled_clustering.append(cluster)

    return scaled_filenames, scaled_clustering


def reduce_dcc_clusters(cluster_data_directory, out_data_dir, k, lr, class_zero_size_multiplier=-1, thresholds=(20,)):
    """
    Main method for limiting DCC-clusters to only values above/equal to a threshold.
    Saved the permitted DCC-clusters to a file, one file for each threshold given

    Parameters
    ----------
    cluster_data_directory      :   str
                                    Path to the directory where cluster data is found
    out_data_dir                :   str
                                    Path to the directory where the output data will be saved
    k                           :   int
                                    Parameter from dcc, used to load the correct cluster data set
    lr                          :   int, float
                                    Parameter from dcc, used to load the correct cluster data set
    class_zero_size_multiplier  :   int, default = -1
                                    When above 0, scales cluster 0 to a factor of the second largest cluster
    thresholds                  :   tuple of ints, default = (20,)
                                    Thresholds based on size of clusters.
                                    A new data set will be made for each value given.

    Returns
    -------
    None
    """

    k = str(k)
    lr = str(lr)
    lr = lr.replace('.', '_')

    clustering = sio.loadmat(os.path.join(cluster_data_directory, 'results/features_k{}_lr{}'.format(k, lr)))
    clustering = clustering['cluster'][0].astype(np.int)

    traindata = sio.loadmat(os.path.join(cluster_data_directory, 'traindata.mat'))
    testdata = sio.loadmat(os.path.join(cluster_data_directory, 'testdata.mat'))
    fulldata = np.concatenate(
        (traindata['filenames'][:], testdata['filenames'][:]),
        axis=0
    )

    img_cluster_path = os.path.join(out_data_dir, r'img_clusters')
    if not os.path.exists(img_cluster_path):
        os.mkdir(img_cluster_path)

    # Starting thresholding
    for threshold in thresholds:
        clustering_above_threshold, _, _ = threshold_clusters(
            clustering=clustering,
            file_names=fulldata,
            threshold=threshold
        )

        clusters = [x[0] for x in clustering_above_threshold]
        filenames = [x[1] for x in clustering_above_threshold]

        clusters = make_cluster_values_adjacent(clusters)

        df = pd.DataFrame({
            'cluster': clusters,
            'filename': filenames,
        })

        if class_zero_size_multiplier > 0:
            scale_cluster_zero(filenames=filenames, clustering=clusters, size_multiplier=class_zero_size_multiplier)
            path_to_csv = os.path.join(
                img_cluster_path,
                r'img_k{}_lr{}_threshold{}_imbalance_degree{}.csv'.format(k, lr, threshold, class_zero_size_multiplier)
            )
        else:
            path_to_csv = os.path.join(img_cluster_path, r'img_k{}_lr{}_threshold{}.csv'.format(k, lr, threshold))

        df.to_csv(path_to_csv)


if __name__ == '__main__':
    k = 30
    lr = 0.1
    imbalance_degree = 4
    thresholds = (10, 20, 30)

    in_data_dir = r'/uio/hume/student-u31/eirikolb/Documents/DCC-master/data/child_poet'
    out_data_dir = r'/uio/hume/student-u31/eirikolb/img'

    reduce_dcc_clusters(
        cluster_data_directory=in_data_dir,
        out_data_dir=out_data_dir,
        k=k, lr=lr,
        thresholds=thresholds,
        class_zero_size_multiplier=imbalance_degree,
    )


