import os
import re
import glob
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

"""
TODO: comment this file
"""


def iterate_image_files(folder_path, label):
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

    dtype = [("env_key", 'U26'), ("pos", float), ("timestamp", float), ("filename", 'U100')]
    array = np.array(list(zip(keys, positions, timestamps, filenames)), dtype=dtype)
    return array


def clean(data, distance_threshold=1):
    data.sort(axis=0, order=("env_key", "pos"))
    prev_env = "none"
    prev_pos = 0
    entries_to_keep = []
    for i in range(data.shape[0]):
        key, pos, _, _ = data[i]
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


def write_to_csv(entries, folder_path, label):
    df = pd.DataFrame(entries, columns=["env_key", "pos", "timestamp", "filename"])
    csv_file_name = folder_path + r'/{}.csv'.format(label)
    df.to_csv(csv_file_name)
    return csv_file_name


def display_images(csv_file, image_folder, wait_for_input=True):
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

        filename, key, pos = df.loc[curr_pos[0], ["filename", "env_key", "pos"]]
        img = plt.imread(image_folder + "/" + filename)

        ax.cla()
        ax.set_title("key: " + key + " | pos: " + str(pos))
        ax.imshow(img, vmin=0, vmax=1)
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
    current_folder_path = r'/uio/hume/student-u31/eirikolb/img/poet_15_nov_24h'
    current_label = "non_obstacle"
    array_of_entries = iterate_image_files(folder_path=current_folder_path, label=current_label)
    cleaned_entries = clean(array_of_entries, distance_threshold=1)
    csv_filename_non_obstacle = write_to_csv(cleaned_entries, folder_path=current_folder_path, label=current_label)

    current_label = "obstacle"
    array_of_entries = iterate_image_files(folder_path=current_folder_path, label=current_label)
    cleaned_entries = clean(array_of_entries, distance_threshold=1)
    csv_filename_obstacle = write_to_csv(cleaned_entries, folder_path=current_folder_path, label=current_label)

    display_images(csv_filename_obstacle, current_folder_path, wait_for_input=True)

