import json
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from pathlib import Path
import logging
from torchvision import transforms

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_image(path):
    """ Load an image from the given path. """
    return np.array(Image.open(path))


def get_patch(image, center, radius):
    """ Extract a patch from the image centered at 'center' with the given 'radius'. """
    x, y = center
    return image[y - radius:y + radius + 1, x - radius:x + radius + 1]


def calculate_difference(patch1, patch2):
    """ Calculate the L2 norm (Euclidean distance) between two patches. """
    difference = patch1 - patch2
    squared_difference = np.square(difference)
    l2_distance = np.sum(squared_difference)

    return l2_distance


def compute_dai(original_image, result_image, points, radius):
    """ Compute the Drag Accuracy Index (DAI) for the given images and points. """
    dai = 0
    for start, target in points:
        original_patch = get_patch(original_image, start, radius)
        result_patch = get_patch(result_image, target, radius)
        dai += calculate_difference(original_patch, result_patch)
    dai /= len(points)
    dai /= cal_patch_size(radius)
    return dai / len(points)


def get_points(points_dir):
    with open(points_dir, 'r') as file:
        points_data = json.load(file)
        points = points_data['points']

    # Assuming pairs of points: [start, target, start, target, ...]
    point_pairs = [(points[i], points[i + 1]) for i in range(0, len(points), 2)]
    return point_pairs


def cal_patch_size(radius: int):
    return (1 + 2 * radius) ** 2


def compute_average_dai(radius, dataset_path, original_dataset_path=None):
    """ Compute the average DAI for a given dataset. """
    dataset_dir = Path(dataset_path)
    original_dataset_dir = Path(original_dataset_path) if original_dataset_path else dataset_dir
    total_dai, num_folders = 0, 0
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    for item in dataset_dir.iterdir():
        if item.is_dir() or (item.is_file() and original_dataset_path):
            folder_name = item.stem if item.is_file() else item.name
            original_image_path = original_dataset_dir / folder_name / 'original.jpg'
            result_image_path = item if item.is_file() else item / 'output_image.png'
            points_json_path = original_dataset_dir / folder_name / 'points.json'

            if original_image_path.exists() and result_image_path.exists() and points_json_path.exists():
                original_image = load_image(str(original_image_path))
                result_image = load_image(str(result_image_path))
                point_pairs = get_points(str(points_json_path))

                original_image = transform(original_image).permute(1, 2, 0).numpy()
                result_image = transform(result_image).permute(1, 2, 0).numpy()
                dai = compute_dai(original_image, result_image, point_pairs, radius)
                total_dai += dai
                num_folders += 1
            else:
                logging.warning(f"Missing files in {folder_name}")

    if num_folders > 0:
        average_dai = total_dai / num_folders
        logging.info(f'Average DAI for {dataset_dir} with r3 {radius} is {average_dai:.4f}. Total {num_folders} images.')
    else:
        logging.warning("No valid folders found for DAI calculation.")


def main():
    gamma = [1, 5, 10, 20]
    result_folder = './bench_result'
    data_folder = './dataset'
    for r in gamma:
        compute_average_dai(r, result_folder, data_folder)


if __name__ == '__main__':
    main()
