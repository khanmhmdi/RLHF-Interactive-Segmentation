import os
import shutil
import random
from tqdm import tqdm


def split_folders(source_dir, destination_dir, train_ratio=0.7, valid_ratio=0.1, test_ratio=0.2):
    """
  Splits folders into train, valid, and test sets.

  Args:
      source_dir: The directory containing the folders to split.
      destination_dir: The directory where the train, valid, and test folders will be created.
      train_ratio: The proportion of folders for the training set.
      valid_ratio: The proportion of folders for the validation set.
      test_ratio: The proportion of folders for the test set.

  Returns:
      A tuple containing the paths to the train, valid, and test directories.
  """

    # Create target directories in the destination folder
    train_dir = os.path.join(destination_dir, 'train')
    valid_dir = os.path.join(destination_dir, 'valid')
    test_dir = os.path.join(destination_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Get a list of all folders in the source directory and shuffle them randomly
    folders = [f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))]
    random.shuffle(folders)

    # Determine split indices based on the provided ratios
    total_folders = len(folders)
    train_split = int(train_ratio * total_folders)
    valid_split = int((train_ratio + valid_ratio) * total_folders)

    # Split folders into train, valid, and test sets
    for i, folder in tqdm(enumerate(folders)):
        source_folder = os.path.join(source_dir, folder)
        if i < train_split:
            target_folder = os.path.join(train_dir, folder)
        elif i < valid_split:
            target_folder = os.path.join(valid_dir, folder)
        else:
            target_folder = os.path.join(test_dir, folder)

        # Move each folder to the corresponding target directory
        shutil.move(source_folder, target_folder)

    # Return the paths to the train, valid, and test directories
    return train_dir, valid_dir, test_dir


# Example usage (commented out)
# source_directory = '/home/kasra/Desktop/New Folder/PosEstimation/data'
# destination_directory = '/home/kasra/Desktop/New Folder/PosEstimation/split_data'
# train_dir, valid_dir, test_dir = split_folders(source_directory, destination_directory)
#
# print(f"Train directory: {train_dir}")
# print(f"Validation directory: {valid_dir}")
# print(f"Test directory: {test_dir}")

import os
from pathlib import Path


def count_files_and_folders(directory):
    """
  Counts the number of files and folders in a given directory and its subdirectories.

  Args:
      directory: The path to the root directory.

  Returns:
      A tuple containing the number of files and the number of folders.
  """
    file_count = 0
    folder_count = 0

    # Walk through the directory tree to count files and folders
    for root, dirs, files in os.walk(directory):
        folder_count += len(dirs)
        file_count += len(files)

    return file_count, folder_count


def rename_labels(root_dir):
    """
  Renames label files to match input image names with a .jpg extension.

  Args:
      root_dir: The path to the root directory containing subfolders.
  """
    # Define the base path and the target path for the processed files
    BASE_PATH = "/home/kasra/Desktop/New Folder/PosEstimation/split_data/valid"
    TRAIN_PATH = '/home/kasra/Desktop/New Folder/PosEstimation/new1/val'

    # Walk through the root directory and its subdirectories
    for subdir, dirs, files in os.walk(root_dir):
        for dir in dirs:
            print(dir)
            # Extract the number from the directory name
            number_of_brats = dir.split("_")[1]

            # Process files in each subdirectory
            for subdir_, dirs_, files_ in os.walk(os.path.join(BASE_PATH, dir)):
                print(files_)
                for file_ in files_:
                    if file_.startswith('t1_') and file_.endswith('.png'):
                        # Copy the T1 image file to the train directory with a new name
                        shutil.copy(os.path.join(BASE_PATH, dir, file_),
                                    os.path.join(TRAIN_PATH, number_of_brats + '_' + file_))

                        # Copy the corresponding segmentation file with a new extension (.jpg)
                        shutil.copy(os.path.join(BASE_PATH, dir, file_.replace('t1', 'seg')),
                                    os.path.join(TRAIN_PATH + "_labels",
                                                 number_of_brats + '_' + file_.replace('.png', '.jpg')))
