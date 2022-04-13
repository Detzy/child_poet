import os
import collections
import matplotlib.pyplot as plt
import obstacle_detector.data_preprocessing as dp


try:
    import cPickle as pickle
except ImportError:
    import pickle


"""
"""


def save_to_file(figure, cluster_label, output_folder_path):
    if not os.path.exists(output_folder_path):
        os.mkdir(output_folder_path)

    figure.savefig(output_folder_path + "/class{}.png".format(cluster_label))


def plot_clustering_samples(clustering, img_filenames, threshold, class_samples, output_folder_path):

    count = collections.Counter(clustering)
    to_plot_from = [k for k in count if count[k] >= threshold]
    print("Number of clusters above threshold {}:".format(threshold), len(to_plot_from))

    for cluster_to_plot in to_plot_from:
        count = 0
        fig = plt.figure()
        # fig = plt.figure(figsize=(10, 10))
        fig.suptitle("Samples from class {}".format(cluster_to_plot))
        for cluster, img_filename in zip(clustering, img_filenames):
            if cluster != cluster_to_plot:
                continue

            img = plt.imread(img_filename)

            fig.add_subplot(3, 4, count + 1)
            plt.imshow(img)
            count += 1
            if count == class_samples:
                save_to_file(fig, cluster_to_plot, output_folder_path)
                plt.close(fig)
                break


def main(img_folder, label_data_path, output_path, threshold, class_samples):
    filenames, labels, number_of_classes = dp.load_cluster_dataset(img_folder_path=img_folder, csv_path=label_data_path)

    plot_clustering_samples(labels, filenames, threshold, class_samples, output_path)


if __name__ == '__main__':
    threshold = 30
    class_samples = 10

    k = '30'
    lr = '0_1'
    img_folder = r'/uio/hume/student-u31/eirikolb/img/poet_dec2_168h/img_files'
    label_data = r'/uio/hume/student-u31/eirikolb/img/img_clusters/img_k30_lr0_1_threshold30_imbalance_degree4.csv'
    out_data_dir = "/uio/hume/student-u31/eirikolb/tmp/class_comparison_k{}_lr{}".format(k, lr)

    main(img_folder, label_data, out_data_dir, threshold, class_samples)


