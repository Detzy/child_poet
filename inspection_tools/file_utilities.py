import glob
import pandas as pd
import pickle


def get_latest_cppn_file(training_run, optimizer_log_file, cppn_genome_folder=None):
    """
    Fetch the file containing the cppn parameters associated with the given optimizer, that were most recently saved

    :param training_run: String, the name of the training run during which the cppn was saved
    :param optimizer_log_file: String, the full path of the optimizer log file belonging to the cppn
    :param cppn_genome_folder: String, full path of the folder containing the cppn genome.
    When None, a default location is used
    """
    if cppn_genome_folder is None:
        cppn_genome_folder = '/uio/hume/student-u31/eirikolb/tmp/niche_encodings/poet_{}/'.format(training_run)

    cppn_genome_file_name = 'genome_*.pickle'
    cppn_genome_file = cppn_genome_folder + cppn_genome_file_name

    # Load the log file corresponding to the optimizer we search for
    optimizer_name = optimizer_log_file.split('/')[-1].split('.')[1]
    log_file = pd.read_csv(optimizer_log_file, sep=',', header=0)

    # All genome keys in the log file should be the same, so we arbitrarily choose the first
    environment_key = log_file['cppn_key_in_{}'.format(optimizer_name)].iloc[0, ]

    assert environment_key == log_file['cppn_key_in_{}'.format(optimizer_name)].iloc[log_file.shape[0]-1, ]

    paired_paths = []
    all_cppn_paths = glob.iglob(cppn_genome_file)
    for cppn_path in all_cppn_paths:
        genome = pickle.load(open(cppn_path, 'rb'))
        if str(genome.key) == str(environment_key):
            paired_paths.append(cppn_path)

    if len(paired_paths) == 0:
        print("Found 0 saved cppns belonging to optimizer {}".format(optimizer_name))
        print("Simulating with base environment")
        return None
    else:
        print("Found {} saved cppns belonging to this optimizer".format(len(paired_paths)))
        print("Returning latest")
        return sorted(paired_paths, reverse=True)[0]


def get_cppn_file_list(training_run, cppn_genome_folder=None):
    """
    Get a list containing all files with cppn parameters in the run

    :param training_run: String, the name of the training run during which the cppns were saved
    :param cppn_genome_folder: String, full path of the folder containing the cppn genomes.
    When None, a default location is used
    """
    if cppn_genome_folder is None:
        cppn_genome_folder = '/uio/hume/student-u31/eirikolb/tmp/niche_encodings/poet_{}/'.format(training_run)

    cppn_genome_file_name = 'genome_*.pickle'
    cppn_genome_file = cppn_genome_folder + cppn_genome_file_name

    return glob.glob(cppn_genome_file)


def get_cppn_file_iterator(training_run, cppn_genome_folder=None):
    """
    Get an iterator containing all files with cppn parameters in the run. Iterator is more memory efficient than list.

    :param training_run: String, the name of the training run during which the cppns were saved
    :param cppn_genome_folder: String, full path of the folder containing the cppn genomes.
    When None, a default location is used
    """
    if cppn_genome_folder is None:
        cppn_genome_folder = '/uio/hume/student-u31/eirikolb/tmp/niche_encodings/poet_{}/'.format(training_run)

    cppn_genome_file_name = 'genome_*.pickle'
    cppn_genome_file = cppn_genome_folder + cppn_genome_file_name

    return glob.iglob(cppn_genome_file)


def get_model_file_list(training_run, model_param_folder=None):
    """
    Get an iterator containing all files with model parameters in the run.

    :param training_run: String, the name of the training run during which the model params were saved
    :param model_param_folder: String, full path of the folder containing the model params.
    When None, a default location is used
    """
    if model_param_folder is None:
        model_param_folder = '/uio/hume/student-u31/eirikolb/tmp/logs/poet_{}/'.format(training_run)

    model_param_file_name = r'poet_*.best.json'
    model_param_file = model_param_folder + model_param_file_name

    return glob.glob(model_param_file)


def get_model_file_iterator(training_run, model_param_folder=None):
    """
    Get an iterator containing all files with model parameters in the run. Iterator is more memory efficient than list.

    :param training_run: String, the name of the training run during which the model params were saved
    :param model_param_folder: String, full path of the folder containing the model params.
    When None, a default location is used
    """
    if model_param_folder is None:
        model_param_folder = '/uio/hume/student-u31/eirikolb/tmp/logs/poet_{}/'.format(training_run)

    model_param_file_name = r'poet_*.best.json'
    model_param_file = model_param_folder + model_param_file_name

    return glob.iglob(model_param_file)
