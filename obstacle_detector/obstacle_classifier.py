import pandas as pd
import tensorflow as tf
import obstacle_detector.data_preprocessing as dp
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt


class ObstacleClassifier:

    def __init__(self, img_height, img_width):
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(img_height, img_width, 1)))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(32, activation='relu'))
        self.model.add(layers.Dense(2))
        self.model.summary()

        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

    def train(self, training_ds, validation_ds):
        history = self.model.fit(training_ds, validation_data=validation_ds, epochs=10)
        return history

    def classify(self, ds_to_classify):
        res = self.model.predict(ds_to_classify)
        print(res)


def get_tvt_datasets(parent_folder_path, labels_to_include=("obstacle", "non_obstacle"),
                     batch_size=10, max_unbalance_degree=1):
    tvt = ["training", "validation", "test"]
    tvt_datasets = []
    for data_type in tvt:
        files, labels = dp.load_dataset_from_csv(
            parent_folder_path=parent_folder_path,
            data_type=data_type,
            labels=labels_to_include,
            max_unbalance_degree=max_unbalance_degree,
        )

        labels = labels.map({label: i for i, label in enumerate(labels_to_include)})

        dataset = tf.data.Dataset.from_tensor_slices((files.values, labels.values))
        tvt_datasets.append(dataset.map(read_image).batch(batch_size))

    training_ds, validation_ds, test_ds = tvt_datasets
    return training_ds, validation_ds, test_ds


def read_image(image_file, label):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_image(image, channels=1, dtype=tf.float32)
    return image, label


if __name__ == "__main__":
    curr_dataset_folder = r'/uio/hume/student-u31/eirikolb/img/poet_18_nov_72h'
    curr_labels = ("obstacle", "non_obstacle")
    curr_batch_size = 10
    unbalance_degree = 2
    training, validation, test = get_tvt_datasets(
        curr_dataset_folder,
        labels_to_include=curr_labels,
        batch_size=curr_batch_size,
        max_unbalance_degree=unbalance_degree
    )

    oc = ObstacleClassifier(32, 32)
    oc.train(training, validation)
    oc.classify(test)
    print(test)

