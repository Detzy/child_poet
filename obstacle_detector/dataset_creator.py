import tensorflow_datasets as tfds


class ObstacleDatasetBuilder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for obstacle dataset."""

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata"""
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict({
                'image': tfds.features.Image(shape=(256, 256, 1)),
                'label': tfds.features.ClassLabel(names=['obstacle', 'non_obstacle']),
            }),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Download the data and define splits."""
        extracted_path = dl_manager.download_and_extract('http://data.org/data.zip')
        # dl_manager returns pathlib-like objects with `path.read_text()`,
        # `path.iterdir()`,...
        return {
            'training': self._generate_examples(path=extracted_path / 'training_images'),
            'validation': self._generate_examples(path=extracted_path / 'validation_images'),
            'test': self._generate_examples(path=extracted_path / 'test_images'),
        }

    def _generate_examples(self, path):
        """Generator of examples for each split."""
        for img_path in path.glob('*.png'):
            # Yields (key, example)
            yield img_path.name, {
                'image': img_path,
                'label': 'obstacle' if img_path.name.startswith('obstacle_') else 'non_obstacle',
            }
