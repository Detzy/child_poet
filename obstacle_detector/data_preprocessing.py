import os
import re
import math
import glob
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def load_dataset_from_csv(parent_folder_path, data_type, labels=("obstacle", "non_obstacle"), max_unbalance_degree=1):
    """
    This method assumes folder given contains one subfolder with the images named /img_files,
    and one folder for each of the data_types "
    """
    seed = 123
    img_folder_path = parent_folder_path + "/img_files"
    csv_folder_path = parent_folder_path + "/{}".format(data_type)

    data_frames = []
    smallest = math.inf
    for label in labels:
        df = pd.read_csv(csv_folder_path + "/{}.csv".format(label))
        smallest = min(df.shape[0], smallest)
        data_frames.append(df)

    data_frames = [df.sample(n=min(smallest*max_unbalance_degree, df.shape[0]), random_state=seed) for df in data_frames]

    full_data_frame = pd.concat(data_frames)

    temp = img_folder_path + "/{}"
    full_data_frame["filename"] = full_data_frame["filename"].map(temp.format)

    return full_data_frame["filename"], full_data_frame["label"]


def iterate_image_files(folder_path, label):
    """
    Iterates through a folder, listing any image file with the given label,
    and produces a custom np.array with file name, environment key, position of death and timestamp.

    :param folder_path: Complete system path of folder containing the images
    :param label: Label of images to include
    :return: np.array of shape (n, 4) where n is the number of images with the given label
    """
    cppn_genome_file = folder_path + r'/{}*.png'.format(label)
    file_iterator = glob.iglob(cppn_genome_file)

    positions = []
    keys = []
    timestamps = []
    filenames = []

    for full_filepath in file_iterator:
        file_name = os.path.basename(full_filepath)
        _, pos, key, timestamp, _ = re.split(label + "_pos|_key|_timestamp|\\.png", file_name)
        positions.append(pos)
        keys.append(key)
        timestamps.append(timestamp)
        filenames.append(file_name)

    labels = [label for i in filenames]
    dtype = [("env_key", 'U26'), ("pos", float), ("timestamp", float), ("filename", 'U100'), ("label", 'U50')]
    array = np.array(list(zip(keys, positions, timestamps, filenames, labels)), dtype=dtype)
    return array


def ensure_location_spacing(data, distance_threshold=1):
    """
    Method to remove images of hurdles that are too close to each other. This should remove overlap of data.

    :param data: np.array from iterate_image_files, containing all image files from a given label
    :param distance_threshold: Images closer together than this will be removed.
    :return: np.array of size (n-k, 4) where n is the number of images with the given label and k is the number of
    images removed because they were too close. 
    """
    data.sort(axis=0, order=("env_key", "pos"))
    prev_env = "none"
    prev_pos = 0
    entries_to_keep = []
    for i in range(data.shape[0]):
        key, pos, _, _, _ = data[i]
        if prev_env != key:
            prev_env = key
        elif prev_pos + distance_threshold < pos:
            pass
        else:
            continue
        prev_pos = pos
        entries_to_keep.append(i)

    data = data[entries_to_keep]
    return data


def split_into_training_validation_test(data, tvt_distribution=(0.70, 0.20, 0.10)):
    """
    Takes in a np.array data set, and splits it into three categories: Training, validation and test.
    
    :param data: np.array with the data that should be split.
    :param tvt_distribution: partition of data between training, validation and test.
    Sum should be 1, and len should be 3
    """

    assert math.fsum(tvt_distribution) == 1
    assert len(tvt_distribution) == 3
    length = data.shape[0]
    training, validation, test = np.split(
        ary=data,
        indices_or_sections=(
            int(tvt_distribution[0] * length),
            int(sum(tvt_distribution[:2]) * length)
        ),
        axis=0
    )
    return training, validation, test


def write_to_csv(data, folder_path, label, tvt_type):
    """
    Write a pandas csv-file containing cleaned and split data with a given label and training/validation/test type.

    :param data: np.array containing the data that should be written to file
    :param folder_path: Full path of the folder in which to save the data
    :param label: The label of the data in :param data:
    :param tvt_type: One of either "training", "validation" and "test".
    :return: The file name of the csv-file saved during this method.
    """
    df = pd.DataFrame(data, columns=["env_key", "pos", "timestamp", "filename", "label"])
    if not os.path.exists(folder_path + r'/{}'.format(tvt_type)):
        os.mkdir(folder_path + r'/{}'.format(tvt_type))
    csv_file_name = folder_path + r'/{}/{}.csv'.format(tvt_type, label)
    df.to_csv(csv_file_name)
    return csv_file_name


def display_images(csv_file, image_folder=None):
    """
    Display the images in a given csv-file! Only shows one image at a time,
    so use keyboard buttons "right" and "left" to navigate, and "escape" to terminate the program.

    :param csv_file: Full path to csv-file containing the images to display
    :param image_folder: Full path to the folder containing the images.
    When None, uses the folder in which the csv-file is located.
    :return: None
    """

    if image_folder is None:
        image_folder = os.path.dirname(csv_file)

    df = pd.read_csv(csv_file)
    curr_pos = [0]

    def key_event(e):
        if e.key == "right":
            curr_pos[0] += 1
        elif e.key == "left":
            curr_pos[0] -= 1
        elif e.key == "escape":
            exit()
        else:
            return

        curr_pos[0] = curr_pos[0] % df.shape[0]

        _filename, _key, _pos = df.loc[curr_pos[0], ["filename", "env_key", "pos"]]
        _img = plt.imread(image_folder + "/" + _filename)

        ax.cla()
        ax.set_title("key: " + _key + " | pos: " + str(_pos))
        ax.imshow(_img, vmin=0, vmax=1)
        fig.canvas.draw()

    fig = plt.figure()
    fig.canvas.mpl_connect('key_press_event', key_event)
    ax = fig.add_subplot(111)

    filename, key, pos = df.loc[curr_pos[0], ["filename", "env_key", "pos"]]
    img = plt.imread(image_folder + "/" + filename)

    ax.set_title("key: " + key + " | pos: " + str(pos))
    ax.imshow(img, vmin=0, vmax=1)
    plt.show()


if __name__ == "__main__":
    current_folder_path = r'/uio/hume/student-u31/eirikolb/img/poet_18_nov_72h'
    unbalance_degree = 4  # Only used for printing information

    # Clean and visualize data for non-obstacles
    current_label = "non_obstacle"
    array_of_entries = iterate_image_files(folder_path=current_folder_path+'/img_files', label=current_label)
    cleaned_entries = ensure_location_spacing(array_of_entries, distance_threshold=1)

    current_training, current_validation, current_test = split_into_training_validation_test(cleaned_entries)
    split_data = {
        "training": current_training,
        "validation": current_validation,
        "test": current_test,
    }
    for data_type in split_data:
        csv_filename = write_to_csv(data=split_data[data_type], folder_path=current_folder_path,
                                    label=current_label, tvt_type=data_type)
        # display_images(csv_filename, image_folder=current_folder_path)

    # Clean and visualize data for non-obstacles
    current_label = "obstacle"
    array_of_entries = iterate_image_files(folder_path=current_folder_path+'/img_files', label=current_label)
    cleaned_entries = ensure_location_spacing(array_of_entries, distance_threshold=1)

    current_training, current_validation, current_test = split_into_training_validation_test(cleaned_entries)
    split_data = {
        "training": current_training,
        "validation": current_validation,
        "test": current_test,
    }
    for data_type in split_data:
        csv_filename = write_to_csv(data=split_data[data_type], folder_path=current_folder_path,
                                    label=current_label, tvt_type=data_type)
        # display_images(csv_filename, image_folder=current_folder_path)

    # Fetch dataset with given unbalance degree and print number of images fetched.
    res = load_dataset_from_csv(current_folder_path, data_type="training", max_unbalance_degree=unbalance_degree)
    print(
        "Dataset with unbalance degree {} gives {} images".format(
            unbalance_degree,
            res[0].shape[0],
        )
    )



