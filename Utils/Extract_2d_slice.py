import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def get_patient_data_paths(dataset_folder):
    """
    Retrieves a list of lists containing paths to BRATS data for each patient
    (assuming individual folders represent patients).

    Args:
        dataset_folder (str): Path to the folder containing the BRATS dataset.

    Returns:
        list: List of lists, where each inner list contains file paths for a patient.
    """
    patient_data_paths = []

    # Iterate over each entry in the dataset folder
    for entry in os.listdir(dataset_folder):
        entry_path = os.path.join(dataset_folder, entry)

        # If the entry is a directory (i.e., a patient folder)
        if os.path.isdir(entry_path):
            patient_files = []

            # Gather all files in the patient's folder
            for file in os.listdir(entry_path):
                file_path = os.path.join(entry_path, file)
                patient_files.append(file_path)

            # Sort files for consistent ordering
            patient_files.sort()
            patient_data_paths.append(patient_files)

    return patient_data_paths


def find_top_slices(mask, axis, value=4, top_n=80):
    """
    Identifies the top slices in a 3D mask array that contain the most of a specified value.

    Args:
        mask (numpy.ndarray): The 3D array (mask) to search through.
        axis (int): The axis along which to search for slices.
        value (int, optional): The value to look for in the mask (default is 4).
        top_n (int, optional): The number of top slices to return (default is 80).

    Returns:
        tuple: A tuple containing:
            - top_slices: Indices of the top slices containing the most of the specified value.
            - random_zero_index: Random indices of slices containing no target value.
    """
    # Count the occurrences of the target value along the specified axis
    count_slices = np.sum(mask == value, axis=(0, 1))

    # Flatten the count array to sort and get top slice indices
    flat_count_slices = count_slices.flatten()
    top_indices = np.argsort(flat_count_slices)[-top_n:][::-1]

    # Find indices of slices with no target value and pick random ones
    zero_indices = np.where(count_slices == 0)[0]
    random_zero_index = np.random.choice(zero_indices[15:-15], size=5)

    # Get the top slice indices
    top_slices = np.unravel_index(top_indices, count_slices.shape)

    return top_slices, random_zero_index


def process_single_patient(source_folder, save_directory):
    """
    Processes data for a single patient, extracting and saving relevant slices
    from the T1 and segmentation files.

    Args:
        source_folder (list): List of file paths related to a single patient.
        save_directory (str): Path to the directory where slices should be saved.
    """
    try:
        # Assume T1 and segmentation paths based on their positions in the list
        t1_path = source_folder[3]
        seg_path = source_folder[1]

        # Load the T1 and segmentation data using nibabel
        t1 = nib.load(t1_path)
        t1 = np.array(t1.dataobj)

        seg = nib.load(seg_path)
        seg = np.array(seg.dataobj)

        # Set non-target values to 0
        seg[seg != 4] = 0

        # Find the top slices and some slices with no target value
        top_slices, least_indices = find_top_slices(seg, 1)

        # Create a directory for saving the slices
        folder_name = os.path.basename(t1_path)
        save_dir = os.path.join(save_directory, folder_name)
        os.makedirs(save_dir, exist_ok=True)

        # Save the top slices as both .npy and .png files
        for slice_index in top_slices[0]:
            t1_slice = t1[:, :, slice_index]
            seg_slice = seg[:, :, slice_index]

            t1_slice_path = os.path.join(save_dir, f't1_{slice_index}.npy')
            seg_slice_path = os.path.join(save_dir, f'seg_{slice_index}.npy')

            np.save(t1_slice_path, t1_slice)
            np.save(seg_slice_path, seg_slice)

            t1_slice_path = os.path.join(save_dir, f't1_{slice_index}.png')
            seg_slice_path = os.path.join(save_dir, f'seg_{slice_index}.png')

            plt.imsave(t1_slice_path, t1_slice, cmap='gray')
            plt.imsave(seg_slice_path, seg_slice, cmap='gray')

        # Save random slices with no target value as both .npy and .png files
        for slice_index in least_indices:
            t1_slice = t1[:, :, slice_index]
            seg_slice = seg[:, :, slice_index]

            t1_slice_path = os.path.join(save_dir, f't1_{slice_index}.npy')
            seg_slice_path = os.path.join(save_dir, f'seg_{slice_index}.npy')

            np.save(t1_slice_path, t1_slice)
            np.save(seg_slice_path, seg_slice)

            t1_slice_path = os.path.join(save_dir, f't1_{slice_index}.png')
            seg_slice_path = os.path.join(save_dir, f'seg_{slice_index}.png')

            plt.imsave(t1_slice_path, t1_slice, cmap='gray')
            plt.imsave(seg_slice_path, seg_slice, cmap='gray')

    except Exception as e:
        print(f"Error processing folder {save_directory}: {e}")


def process_patient_data(patient_data_paths, save_directory, max_workers=4):
    """
    Processes data for all patients concurrently, saving relevant slices.

    Args:
        patient_data_paths (list): List of lists, where each inner list contains file paths for a patient.
        save_directory (str): Path to the directory where slices should be saved.
        max_workers (int, optional): Maximum number of threads to use for processing (default is 4).
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit each patient's data for concurrent processing
        futures = [executor.submit(process_single_patient, folder, save_directory) for folder in patient_data_paths]

        # Use tqdm to visualize the progress of the processing
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                future.result()  # Ensure that any exceptions are raised
            except Exception as e:
                print(f"Error in future: {e}")
