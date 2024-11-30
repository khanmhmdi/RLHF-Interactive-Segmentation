import os
import shutil
from tqdm import tqdm

import os
import shutil
from concurrent.futures import ThreadPoolExecutor


def copy_and_rename_files(dir, root_dir, destination_path):
    BASE_PATH = root_dir
    TRAIN_PATH = destination_path
    number_of_brats = dir.split("_")[1]

    for subdir_, dirs_, files_ in os.walk(os.path.join(BASE_PATH, dir)):
        for file_ in files_:
            if file_.startswith('t1_') and file_.endswith('.png'):
                shutil.copy(
                    os.path.join(BASE_PATH, dir, file_),
                    os.path.join(TRAIN_PATH, number_of_brats + '_' + file_)
                )
                shutil.copy(
                    os.path.join(BASE_PATH, dir, file_.replace('t1', 'seg')),
                    os.path.join(TRAIN_PATH + "_labels", number_of_brats + '_' + file_.replace('.png', '.jpg'))
                )


def rename_labels(root_dir, destination_path):
    with ThreadPoolExecutor() as executor:
        for subdir, dirs, files in os.walk(root_dir):
            # Submit tasks to the ThreadPoolExecutor
            executor.map(lambda dir: copy_and_rename_files(dir, root_dir, destination_path), dirs)
