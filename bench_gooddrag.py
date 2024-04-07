# *************************************************************************
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# *************************************************************************


import os
import sys
import cv2
import numpy as np
import json
from pathlib import Path
from PIL import Image
from utils.ui_utils import run_gooddrag, train_lora_interface, show_cur_points, create_video


def benchmark_dataset(dataset_folder):
    dataset_path = Path(dataset_folder)
    subfolders = [f for f in dataset_path.iterdir() if f.is_dir() and f.name != '.ipynb_checkpoints']

    for subfolder in subfolders:
        print(f'Benchmarking {subfolder.name}...')
        try:
            bench_one_image(subfolder)
        except Exception as e:
            print(f'An error occured while benchmarking {subfolder.name}: {e}.')


def load_data(folder):
    """Load the original image, mask, and points from the specified folder."""
    folder_path = Path(folder)

    # Load original image
    original_image_path = folder_path / 'original.jpg'
    original_image = Image.open(original_image_path)
    original_image = np.array(original_image)

    # Load mask
    mask_path = folder_path / 'mask.png'
    mask = Image.open(mask_path)
    mask = np.array(mask)
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]

    # Load points
    points_path = folder_path / 'points.json'
    with open(points_path, 'r') as f:
        points_data = json.load(f)
        points = points_data['points']

    image_points_path = folder_path / 'image_with_points.jpg'
    image_with_points = Image.open(image_points_path)
    image_with_points = np.array(image_with_points)

    return original_image, mask, points, image_with_points


def bench_one_image(folder):
    """
    Test the saved data by running the drag model.

    Args:
      folder: The folder where the original image, mask, and points are saved.
    """
    original_image, mask, points, image_with_points = load_data(folder)
    model_path = 'runwayml/stable-diffusion-v1-5'

    lora_path = f'./lora_data/{folder.parts[-1]}'

    print(f'Training Lora.')
    train_lora_interface(original_image=original_image, prompt='', model_path=model_path,
                         vae_path='stabilityai/sd-vae-ft-mse',
                         lora_path=lora_path, lora_step=70, lora_lr=0.0005, lora_batch_size=4, lora_rank=16,
                         use_gradio_progress=False)
    print(f'Training Lora Done! Begin dragging.')

    return_intermediate_images = True

    result_dir = f'./bench_result/{Path(folder).parts[-1]}'
    os.makedirs(result_dir, exist_ok=True)

    output_image, new_points = run_gooddrag(
        source_image=original_image,
        image_with_clicks=image_with_points,
        mask=mask,
        prompt='',
        points=points,
        inversion_strength=0.75,
        lam=0.1,
        latent_lr=0.02,
        model_path=model_path,
        vae_path='stabilityai/sd-vae-ft-mse',
        lora_path=lora_path,
        drag_end_step=7,
        track_per_step=10,
        save_intermedia=False,
        compare_mode=False,
        r1=4,
        r2=12,
        d=4,
        max_drag_per_track=3,
        drag_loss_threshold=0,
        once_drag=False,
        max_track_no_change=5,
        return_intermediate_images=return_intermediate_images,
        result_save_path=result_dir
    )

    print(f'Drag finished!')
    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
    output_image_path = os.path.join(result_dir, 'output_image.png')
    cv2.imwrite(output_image_path, output_image)

    img_with_new_points = show_cur_points(np.ascontiguousarray(output_image), new_points, bgr=True)
    new_points_image_path = os.path.join(result_dir, 'image_with_new_points.png')
    cv2.imwrite(new_points_image_path, img_with_new_points)

    points_path = os.path.join(result_dir, f'new_points.json')
    with open(points_path, 'w') as f:
        json.dump({'points': new_points}, f)

    if return_intermediate_images:
        create_video(result_dir, folder)


def main(dataset_folder):
    benchmark_dataset(dataset_folder)


if __name__ == '__main__':
    dataset = sys.argv[1]
    main(dataset)
