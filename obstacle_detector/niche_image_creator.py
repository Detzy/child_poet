import glob
import time
import numpy as np
import matplotlib.pyplot as plt
from poet_distributed.niches.box2d.cppn import CppnEnvParams

# Define sizes used to determine cppn input
# to ensure same input as when cppn was trained.
# Taken from bipedal_walker_custom.py

SCALE = 30.0   # affects how fast-paced the game is, forces should be adjusted as well

MOTORS_TORQUE = 80
SPEED_HIP = 4
SPEED_KNEE = 6
LIDAR_RANGE = 160 / SCALE

INITIAL_RANDOM = 5

HULL_POLY = [
    (-30, +9), (+6, +9), (+34, +1),
    (+34, -8), (-30, -8)
]
LEG_DOWN = -8 / SCALE
LEG_W, LEG_H = 8 / SCALE, 34 / SCALE

VIEWPORT_W = 600
VIEWPORT_H = 400

TERRAIN_STEP = 14 / SCALE
TERRAIN_LENGTH = 200     # in steps
TERRAIN_HEIGHT = VIEWPORT_H / SCALE / 4
TERRAIN_GRASS = 10    # low long are grass spots, in steps
TERRAIN_STARTPAD = 20    # in steps
STARTPAD_LENGTH = TERRAIN_STEP * TERRAIN_STARTPAD
FRICTION = 2.5
MID = TERRAIN_LENGTH * TERRAIN_STEP / 2.


class NicheImageCreator:
    """
    Creates and collects images based on encoded environments from CPPN.
    Images are centered around points of interest, determined from some sort of event.
    Events vary in type, for example the death of an agent, an agent moving unusually slowly,
    or a random decision to select a place where an agent did well.
    Images are then labeled, based on what type of event triggered the image creation.
    """

    def __init__(self, cppn_params=None, dataset_folder=None):
        """
        :param cppn_params: full file path to saved cppn parameters,
        or an instance of a CppnEnvParams object
        :param dataset_folder: absolute path of the location to save dataset images
        """
        self.cppn_params = cppn_params
        self.dataset_folder = dataset_folder
        self.current_image = None

    @property
    def dataset_folder(self):
        """
        Folder where images for datasets are saved
        """
        if self._dataset_folder is not None:
            return self._dataset_folder
        else:
            print("No folder has been set as dataset location")

    @dataset_folder.setter
    def dataset_folder(self, folder):
        self._dataset_folder = folder

    @property
    def current_image(self):
        """
        Most recently produced image
        """
        if self._current_image is not None:
            return self._current_image
        else:
            print("No image has been made")

    @current_image.setter
    def current_image(self, image):
        self._current_image = image

    @property
    def cppn_params(self):
        """
        CPPN that encodes a niche environment
        """
        return self._cppn_params

    @cppn_params.setter
    def cppn_params(self, cppn):
        if isinstance(cppn, CppnEnvParams):
            self._cppn_params = cppn
        else:
            self._cppn_params = CppnEnvParams(genome_path=cppn)

    def altitude_function(self, x):
        transformed_x = (x - MID) * np.pi / MID
        return self._cppn_params.altitude_fn((transformed_x,))[0]

    def create_image(self, mid_x, in_width=8, in_height=8, out_width=40, out_height=40):
        """
        Draws a 1 channel binary image around a point in an environment niche, based on a cppn encoding.
        The cppn encodes the coordinate of the separation between sky and ground in a niche.
        The method outputs a binary image where ground is 1 and sky is 0.
        :param mid_x: The x coordinate of the center point, given in the coordinate space of the niche
        :param in_width: width of the section around the mid-point that should be drawn
        :param in_height: height of the section around the mid-point that should be drawn
        :param out_width: width of the output image
        :param out_height: height of the output image
        :return: np.array of size (height, width)
        """
        # The thought process behind this method is described here:
        # We define x,y as coordinates in the input image space, and i,j as coordinates in the output image space.
        # Then we define f as the altitude_function that the cppn encodes, such that f(x) = y
        # Finally, we define g(i) = x, and h(y) = j.
        # This lets us find j given an i like this: j = h(f(g(i)))

        # Since the bipedal walker has a flat start pad, regions before the default start pad length are also flat
        # in this encoded image

        # Define functions and variables
        startpad_height = self.altitude_function(STARTPAD_LENGTH+TERRAIN_STEP)

        mid_y = self.altitude_function(mid_x) if mid_x > STARTPAD_LENGTH+TERRAIN_STEP else startpad_height
        min_x, min_y = mid_x - in_width/2, mid_y - in_height/2
        max_x, max_y = mid_x + in_width/2, mid_y + in_height/2

        min_i, min_j = 0, 0
        max_i, max_j = out_width, out_height

        def i_to_x(i_):
            i_fraction = (i_ - min_i) / (max_i - min_i)
            return i_fraction * (max_x - min_x) + min_x

        def y_to_j(y_):
            y_fraction = (y_ - min_y) / (max_y - min_y)
            return y_fraction * (max_j - min_j) + min_j

        # Make the image
        out_image = np.zeros((out_width, out_height), dtype=float)

        for i in range(out_width):
            # Loop through every i of the out_image, and find the corresponding j,
            # through x,y pairs
            x = i_to_x(i)
            y = self.altitude_function(x) if x > STARTPAD_LENGTH+TERRAIN_STEP else startpad_height
            j = y_to_j(y)

            # we need discrete values for indexing. This could potentially be done by some form of cross-sampling
            rounded_j = round(j)
            rounded_j = max(0, min(out_height, rounded_j))
            out_image[i, :rounded_j] = np.ones((rounded_j,), dtype=float).T

        # Because a numpy arrays have axes from top left, we need to rotate 90 degrees
        out_image = np.rot90(out_image)
        self.current_image = out_image
        return out_image

    def save_image_for_dataset(self, image_to_save, class_label, x_pos, cppn_key):
        time_of_saving = time.time()
        filename = self.dataset_folder + "/{}_pos{}_key{}_timestamp{}.png".format(class_label, x_pos,
                                                                                  cppn_key, time_of_saving)
        plt.imsave(filename, image_to_save, vmin=0, vmax=1)

    def save_image(self, filename):
        plt.imsave(filename, self.current_image, vmin=0, vmax=1)

    def show_image(self):
        plt.imshow(self.current_image, vmin=0, vmax=1)
        plt.show()


if __name__ == "__main__":
    """
    Just some code for testing. This file should rarely be used as main
    """
    label = "test"
    identifier = "temp"
    save_to = r'{}_{}.png'.format(label, identifier)

    test_run_name = 'sep30_overnight'
    cppn_genome_folder = '/uio/hume/student-u31/eirikolb/tmp/niche_encodings/poet_{}/'.format(test_run_name)
    # cppn_genome_file_name = 'genome_{}.pickle'.format(time_string)  # test just one
    cppn_genome_file_name = 'genome_*.pickle'  # loop all
    cppn_genome_file = cppn_genome_folder + cppn_genome_file_name

    for cppn_genome in glob.iglob(cppn_genome_file):
        t = NicheImageCreator(cppn_genome)
        t.current_image = t.create_image(400, 255, 255)
        t.show_image()

        # t.save_image(save_to)
