import tensorflow as tf
import obstacle_detector.data_preprocessing as dp
from tensorflow.keras import layers, models
import mlflow as mlf
import numpy as np

DEFAULT_MODEL = r'cnn_models/default_model'


class ObstacleClassifier:

    def __init__(self, img_height=32, img_width=32, n_classes=2, cp_callback=None):
        """

        Parameters
        ----------
        img_height  :   int, default=32
                        Height of the input images
        img_width   :   int, default=32
                        Width of the input images
        n_classes   :   int, default=1
                        Number of classes for the training data
        cp_callback :   ModelCheckpoint object, optional
                        Model Checkpoint from tf.keras.callbacks, used to log the weights of the model during training
        """
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(filters=8, kernel_size=(3, 3), activation='relu', input_shape=(img_height, img_width, 1)))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(8, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(8, (3, 3), activation='relu'))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(units=16, activation='relu'))
        self.model.add(layers.Dense(units=n_classes))
        self.model.summary()

        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

        self.cp_callback = cp_callback

    def train(self, n_epochs, training_ds, validation_ds=None):

        callback = [self.cp_callback] if self.cp_callback is not None else None

        if validation_ds is None:
            history = self.model.fit(
                training_ds,
                epochs=n_epochs,
                validation_split=0.2,
                callbacks=callback)
        else:
            history = self.model.fit(
                training_ds,
                validation_data=validation_ds,
                epochs=n_epochs,
                callbacks=callback)
        return history

    def evaluate(self, ds_to_evaluate_on):
        res = self.model.evaluate(ds_to_evaluate_on)
        return res

    def classify(self, ds_to_classify):
        res = self.model.predict(ds_to_classify)
        return res

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model = tf.keras.models.load_model(path)
        print(self.model.summary())

    def resume_from_checkpoint(self, path):
        self.model.load_weights(path)
        print(self.model.summary())


def get_dcc_dataset(img_folder, label_data_path, ttv_split=(0.7, 0.2), batch_size=10, max_imbalance_degree=-1):
    """
    Loads an image dataset, clustered by DCC, from a csv file. Converts the dataset to one hot encoding,
    and wraps it in the tensorflow Dataset class. The dataset is split into training, validation and test data.

    Parameters
    ----------
    img_folder              : str
                              Path to the folder containing the images of the dataset
    label_data_path         : str
                              Path to the file containing the DCC labels of the dataset
    ttv_split               : tuple of float, default=(0.7, 0.2)
                              Tuple of length 2 containing the fraction of the dataset assigned to training data,
                              and then the fraction of the dataset assigned to validation data.
                              The remaining fraction will be test data.
    batch_size              : int, default=10
                              Batch size of the output Dataset-objects
    max_imbalance_degree    : int, default=-1
                              Imbalance degree between largest and second largest class.
                              NOTE: THIS IS NOT YET IMPLEMENTED, ANY VALUE OTHER THAN -1 WILL RAISE ERROR

    Returns
    -------
    training_ds         : tf.Dataset
    validation_ds       : tf.Dataset
    test_ds             : tf.Dataset
    number_of_classes   : int
                          The number of classes in the labeled data set
    """
    assert sum(ttv_split) <= 1

    filenames, labels, number_of_classes = dp.load_cluster_dataset(img_folder_path=img_folder, csv_path=label_data_path,
                                                                   max_imbalance_degree=max_imbalance_degree)

    labels = tf.one_hot(labels.values, number_of_classes)

    dataset = tf.data.Dataset.from_tensor_slices((filenames.values, labels))
    dataset = dataset.map(read_image).batch(batch_size)

    train_size = int(ttv_split[0] * len(dataset))
    validation_size = int(ttv_split[1] * len(dataset))

    training_ds = dataset.take(train_size)
    validation_ds = dataset.skip(train_size).take(validation_size)
    test_ds = dataset.skip(train_size + validation_size)

    return training_ds, validation_ds, test_ds, number_of_classes


def read_image(image_file, label):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_image(image, channels=1, dtype=tf.float32)
    return image, label


def main(img_folder, label_file, checkpoint_save_path, model_save_path, mlf_runs_save_path,
         experiment_name=None, batch_size=250, epochs=(100,)):
    mlf.set_tracking_uri(mlf_runs_save_path)
    if experiment_name is not None:
        mlf.create_experiment(name=experiment_name)

    results = []

    for e in epochs:
        train_data, val_data, test_data, n_classes = get_dcc_dataset(
            img_folder,
            label_data_path=label_file,
            batch_size=batch_size
        )

        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_save_path.format(e),
            save_weights_only=True,
            verbose=1,
        )

        oc = ObstacleClassifier(32, 32, n_classes=n_classes, cp_callback=cp_callback)

        mlf.tensorflow.autolog(every_n_iter=1)

        oc.train(n_epochs=e, training_ds=train_data, validation_ds=val_data)
        test_result = oc.evaluate(test_data)

        results.append(test_result)
        oc.save_model(model_save_path+r'/{}epochs_{}test_score'.format(e, test_result[1]))
        mlf.end_run()

    for e, r in zip(epochs, results):
        print("Epochs: {} | Test result: {}".format(e, r[1]))


if __name__ == "__main__":
    # Inputs
    img_folder = r'/uio/hume/student-u31/eirikolb/img/poet_dec2_168h/img_files'
    label_data = \
        r'/uio/hume/student-u31/eirikolb/img/img_clusters/img_k30_lr0_1_threshold30_imbalance_degree4_man_relabel.csv'
    # label_data = r'/uio/hume/student-u31/eirikolb/img/img_clusters/img_k30_lr0_1_threshold30.csv'

    # Outputs/saves
    checkpoint_path = r'/uio/hume/student-u31/eirikolb/tmp/cnn_logs/{}epochs_cp.ckpt'
    model_path = r'/uio/hume/student-u31/eirikolb/Documents/child_poet/cnn_models'
    mlf_runs = r'file:/uio/hume/student-u31/eirikolb/Documents/child_poet/mlruns'

    curr_batch_size = 250
    epochs = (30, 50, 80, 100, 150, 200, 250, 300, 350, 400, 450)
    experiment_name = 'Manual Relabeling'
    # epochs = (250, 300, 350, 400, 450)

    main(
        img_folder, label_data, checkpoint_path, model_path, mlf_runs,
        experiment_name=experiment_name, batch_size=curr_batch_size, epochs=epochs
    )

