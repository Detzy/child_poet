from collections import Iterator

import tensorflow_datasets as tfds

"""
HMMM, vet ikke om dette er noe vitts?
"""


class ObstacleDatasetBuilder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for obstacle dataset."""

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata"""
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict({
                'image': tfds.features.Image(shape=(40, 40, 1)),
                'label': tfds.features.ClassLabel(names=['obstacle', 'non_obstacle']),
            }),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Assumes the files are already split into three types, training, validation and test"""
        image_file_paths = "/home/dataset/or/something"

        return {
            'training': self._generate_examples(path=image_file_paths / 'training_images'),
            'validation': self._generate_examples(path=image_file_paths / 'validation_images'),
            'test': self._generate_examples(path=image_file_paths / 'test_images'),
        }

    def _generate_examples(self, path):
        """Generator of examples for each split."""
        for img_path in path.glob('*.png'):
            # Yields (key, example)
            yield img_path.name, {
                'image': img_path,
                'label': 'obstacle' if img_path.name.startswith('obstacle') else 'non_obstacle',
            }
