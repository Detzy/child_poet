import glob
import os
import matplotlib.pyplot as plt
import functools
import re
import numpy as np


def plots_prediction_versus_simulation(input_directory, output_directory, file_name_patterns, data_point_threshold=21):
    """
    Function that finds pairs of files in input_directory matching a name in file_name_patterns,
    and plots the values of the first file against the second. The first column is discarded, the second column
    is the values of the y-axis, and the third column is the values of the x-axis.

    Parameters
    ----------
    input_directory     : str
    output_directory    : str
    file_name_patterns  : list of str

    Returns
    -------
    None
    """
    for file_name_pattern in file_name_patterns:
        # Find all files in the input directory that match the file_name_pattern,
        file_names = glob.glob(input_directory + file_name_pattern.format('predicted'))
        for predicted_file in file_names:
            real_file = predicted_file.replace('predicted', 'real')

            # Load the data from the files, and plot the data
            with open(predicted_file, 'r') as f:
                predicted_file_data = f.readlines()
                predicted_file_data = [line.split() for line in predicted_file_data]

            if len(predicted_file_data) < data_point_threshold:
                continue

            with open(real_file, 'r') as f:
                real_file_data = f.readlines()
                real_file_data = [line.split() for line in real_file_data]

            predicted_data = [float(line[1]) for line in predicted_file_data]
            real_file_data = [float(line[1]) for line in real_file_data]
            steps = [int(line[2]) for line in predicted_file_data]

            plt.clf()
            plt.plot(steps, predicted_data, label='Predicted')
            plt.plot(steps, real_file_data, label='Real')
            plt.xlabel('Step')

            # Find the part of the file_name_pattern that matches 'simulation_*_', and use it to label the y-axis
            y_axis_label = re.search('simulation_.*_', file_name_pattern).group(0)
            y_axis_label_with_space = y_axis_label.replace('_', ' ')
            plt.ylabel(y_axis_label_with_space)

            plt.legend()
            plt.title(predicted_file.split('_')[-1])
            ax = plt.gca()
            if y_axis_label_with_space.count('score') > 0:
                ax.set_ylim([-100, 400])
            else:
                ax.set_ylim([0, 100])

            # Save the plot to the output directory
            plt.savefig(output_directory + y_axis_label + predicted_file.split('_')[-1] + '.png')


def rewrite_mlflow_metric_steps(directory):
    """
    Function that loads all files in a directory, changes the third value on every line to an increasing number,
    and saves the result in the same file. This lets us use the mlflow step-option in plots.
    """
    # Load all files in the directory
    files = glob.glob(directory + '/*')
    # For every file
    for file in files:
        # set j = 0
        j = 0
        # Load the file
        with open(file, 'r') as f:
            # Read the file
            lines = f.readlines()
            # For every line
            for i in range(len(lines)):
                # Split the line
                line = lines[i].split(' ')

                # If the line is not empty
                if line[0] != '':
                    # Change the third value to 1
                    line[2] = str(j) + '\n'
                    j += 1
                    # Join the line
                    line = ' '.join(line)
                    # Save the line
                    lines[i] = line
            # Save the file
            with open(file, 'w') as f:
                # print(lines)
                f.writelines(lines)


# Main function that generates plots
def main(input_directory, output_directory):
    # Create the output directory if it does not exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    file_name_patterns = [
        '{}_simulation_score_*',
        '{}_simulation_distance_*',
        ]
    plots_prediction_versus_simulation(input_directory, output_directory, file_name_patterns)


# If the script is run directly, run the main function with a test directory as input
if __name__ == '__main__':
    base_mlflow_directory = '/uio/hume/student-u31/eirikolb/tmp/mlruns/0'
    run_directory = '/9e5a4f0afb714a6ba70f6283c2ba6b9a'
    metrics_directory = '/metrics'
    full_metric_path = base_mlflow_directory + run_directory + metrics_directory + '/'

    output_base_directory = '/uio/hume/student-u31/eirikolb/tmp/plots'
    output_directory = output_base_directory + run_directory + '/'

    main(input_directory=full_metric_path, output_directory=output_directory)
