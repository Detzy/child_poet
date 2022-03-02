from obstacle_detector.obstacle_library import ObstacleLibrary
from obstacle_detector.obstacle_classifier import ObstacleClassifier
from obstacle_detector.niche_image_creator import NicheImageCreator
from inspection_tools.inspect_stumble_detector import generate_novel_cppn_environments


def full_obstacle_library_plotter(cppn_genome_path, n_images_to_show=20):
    """
    This is an inspection tool associated with the ObstacleLibrary.
    Currently, it is very bare bones, used only for quick inspection.
    """
    model_to_load = \
        r'/uio/hume/student-u31/eirikolb/Documents/child_poet/cnn_models/200epochs_0.992277204990387test_score'

    img_creator = NicheImageCreator(cppn_genome_path)
    obstacle_classifier = ObstacleClassifier()
    obstacle_classifier.load_model(model_to_load)

    ol = ObstacleLibrary(img_creator=img_creator, img_classifier=obstacle_classifier)
    ol.display_images(number_of_images_to_show=n_images_to_show)


def main(cppn_genome_path):
    full_obstacle_library_plotter(cppn_genome_path)


if __name__ == "__main__":
    """
    Generate a set of novel cppn encodings, then run some list of agents on these.
    If an agent dies, the location of its stumble is drawn
    """
    test_run_name = 'sep30_overnight'
    test_seeds = [12]

    for test_seed in test_seeds:
        novel_environments = generate_novel_cppn_environments(n_environments=1, n_mutations=50, seed=test_seed)
        for current_cppn_genome in novel_environments:
            main(cppn_genome_path=current_cppn_genome)
