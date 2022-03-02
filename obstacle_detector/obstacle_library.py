import numpy as np
import matplotlib.pyplot as plt
from poet_distributed.niches.box2d.bipedal_walker_custom import TERRAIN_LENGTH, TERRAIN_GRASS, TERRAIN_STEP


class ObstacleLibrary:
    """
    Class for detecting, classifying and containing all obstacles in an environment.
    """

    def __init__(self, img_creator, img_classifier, in_width=8, in_height=8, out_width=32, out_height=32):
        """
        Class keeping track of what type of obstacles are found withing a niche/environment.

        Parameters
        ----------
        img_creator     :   ImageCreator
                            Object for creating images of obstacles in the terrain.
        img_classifier  :   ObstacleClassifier
                            Pretrained CNN for classifying the images from img_creator
        in_width        :   int, default=8
                            Parameter for the img_creator
        in_height       :   int, default=8
                            Parameter for the img_creator
        out_width       :   int, default=32
                            Parameter for the img_creator
        out_height      :   int, default=32
                            Parameter for the img_creator
        """
        self._img_creator = img_creator
        self._img_classifier = img_classifier

        self._in_width = in_width
        self._in_height = in_height
        self._out_width = out_width
        self._out_height = out_height

        terrain_length = (TERRAIN_LENGTH - TERRAIN_GRASS) * TERRAIN_STEP
        terrain_sample_number = 100  # determines the number of images produced from the environment

        self.obstacle_classes = []
        self.label_certainty = []
        self.obstacle_positions = [pos for pos in np.linspace(0, terrain_length, num=terrain_sample_number)]
        self.images = []
        self._classify_terrain()

    def _classify_terrain(self):
        """
        Classifies the terrain associated with this library, and places the classes, classification certainty and
        images in lists.

        Returns
        -------
        None
        """
        images = []
        for sample_position in self.obstacle_positions:
            images.append(self._img_creator.create_image(
                mid_x=sample_position,
                in_width=self._in_width, in_height=self._in_height,
                out_width=self._out_width, out_height=self._out_height,
            ))

        self.images = images

        numpy_images = np.array(images).reshape((len(images), 32, 32, 1))
        classifications = self._img_classifier.classify(numpy_images)

        for c in classifications:
            c = np.array(c)
            self.label_certainty.append(max(c))
            self.obstacle_classes.append(np.argmax(c))

    def display_images(self, number_of_images_to_show=None):
        """
        Just a simple tool to display all classified images.
        Impractical and slightly useless. Be warned.
        Parameters
        ----------
        number_of_images_to_show    :   int, optional
                                        Number of images to show. If None, shows all images.
        """
        for i, image in enumerate(self.images):
            if number_of_images_to_show is not None and i >= number_of_images_to_show:
                break
            fig = plt.figure()
            title = "{} | class given: {}".format(i, self.obstacle_classes[i])
            fig.suptitle(title)
            plt.imshow(image)

        plt.show()
