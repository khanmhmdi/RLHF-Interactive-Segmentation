#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from unsegqc.model import ProbModel
from pycocotools.coco import COCO
import SimpleITK as sitk
import pandas as pd
import multiprocessing as mp
import argparse

# Settings

config = {
    "dim": 2,                           # Image dimension
    "tol": 1e-5,                        # Threshold for convergence
    "eps": 1e-8,
    "n_iter": 500,                      # Maximum number of iterations

    # Parameters of the Student't mixtures
    "n_comp": 7,                        # Number of components for the background an foreground regions
    "SMM_tol": 1e-6,
    "SMM_n_iter": 1000,
    "SMM_min_covar": 1e-6,

    # Parameters for cropping the image
    "use_narrow_band": True,            # If we want to work on a narrow band along the input segmentation border
    "margin": 40,                       # Amount of voxels left around the segmentation when cropping the image
    "bound": 15,                         # Radius of the narrow band

    # Type of spatial regularization
    "regularization_method": "FDSP",      # Type of regularization, can be 'GLSP', 'MRF' or 'FDSP'
}

config_fdsp = {
    # Parameters for FDSP regularization
    "prop": 0.7,                        # To increase the stability, only a fraction of the weights W are updated at
                                        # each iteration
    "alpha0": 1,                        # Initial value of the hyperparameter
}

config_glsp = {
    # Parameters for the GLSP regularization
    "spatial_prior_params": {"istep": 6, "jstep": 6, "iscale": 17, "jscale": 17, "layout": "square"}, # Layout of the basis functions
    "nr_n_iter": 25,                    # Maximum umber of iterations for the Newton-Raphson step
    "nr_tol": 1e-5,                     # Convergence threshold for the Newton-Raphson step
    "alpha0": 1e-3,                     # Initial value of the hyperparameter
}

config_mrf = {
    # Parameters for the MRF regularization
    "beta": 1,                          # Initial value of the hyperparameter
    "connectivity": 1,                  # Radius of the neighborhood taken into account
}


def check_seg(coco, annotation_id, dataPath):

    ann = coco.loadAnns(annotation_id)[0]

    # Load the image
    img = coco.loadImgs(ann["image_id"])[0]
    img = sitk.ReadImage(dataPath+img['file_name'])
    spacing = img.GetSpacing()
    img = sitk.GetArrayFromImage(img)

    # Load the segmentation
    gt = coco.annToMask(ann)
    gt = gt.astype(np.int)

    # Learn the probabilistic model
    prob_model = ProbModel(config)
    prob_model.assess_seg_quality(gt, np.transpose(img, (2, 0, 1)), spacing)

    df = np.array([[ann["image_id"], annotation_id, prob_model.ase, prob_model.dice_score]])
    df = pd.DataFrame(df, columns=["image_id", "annotation_id", "ASE", "Dice"])

    return df


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-coco_dir", help="path to the COCO directory (Example: home/Documents/coco)", type=str)
    parser.add_argument("-regularization_method", help="regularization method ('GLSP', 'FDSP', or 'MRF')", type=str)
    args = parser.parse_args()

    # We use the COCO dataset as an example. We focus on segmentations belonging to the 'bear' category. For each
    # segmentation S inside the dataset, we build a probabilistic segmentation M. Comparison of the adequacies between
    # S and M allows to build a ranking of the segmentations inside the dataset.

    # Update the config dictionary depending on the chosen regularization
    regularization_method = args.regularization_method # Type of regularization, can be 'GLSP', 'MRF' or 'FDSP'
    config["regularization_method"] = regularization_method
    config.update({"FDSP":config_fdsp, "GLSP":config_glsp, "MRF":config_mrf}[regularization_method])

    # Load the annotations
    dataType = 'val2017'
    annFile = '{}/annotations/instances_{}.json'.format(args.coco_dir, dataType)
    dataPath = args.coco_dir+"/"+dataType+"/"
    coco = COCO(annFile)
    annotations = coco.getAnnIds(catIds=coco.getCatIds(catNms=['bear']), iscrowd=None)

    # Loop on the annotations
    results = []
    pool = mp.Pool()
    for annotation_id in annotations:
        r = pool.apply_async(check_seg, args=(coco, annotation_id, dataPath))
        results.append(r)
    for k in range(len(results)):
        results[k] = results[k].get()

    results = pd.concat(results)
    results = results.sort_values(by="ASE", ascending=True)

    # Plot the ASE distribution
    fig = plt.figure(figsize=[4, 4])
    ax = plt.subplot(111)
    n, bins, patches = ax.hist(results["ASE"],
                               bins="auto",
                               facecolor="#0000d5",
                               alpha=1,
                               density=True)
    ax.tick_params(labelsize=10)
    plt.ylabel("Density", size=12)
    plt.xlabel("Average Surface Error", size=12)
    plt.grid(False)
    plt.savefig("histogram.png", bbox_inches='tight', pad_inches=0.1)
    plt.show()

    nrows = 2
    ncols = 3
    count = 0
    # Show well explained cases
    fig = plt.figure(figsize=[4 * ncols, 2.5 * nrows], constrained_layout=True)
    for k in range(nrows * ncols):
        case = results.iloc[k, :]
        img = sitk.GetArrayFromImage(sitk.ReadImage(dataPath + coco.loadImgs(int(case["image_id"]))[0]['file_name']))
        gt = coco.annToMask(coco.loadAnns(int(case['annotation_id']))[0])
        count += 1
        plt.subplot(nrows, ncols, count)
        plt.imshow(img)
        plt.contour(gt, levels=[0.5], colors=["#ffff00"], linewidths=[2])
        plt.axis("off")
    plt.suptitle("Well explained cases", size=16)
    plt.savefig("well_explained_cases.png", bbox_inches='tight', pad_inches=0.1)
    plt.show()

    nrows = 2
    ncols = 3
    count = 0
    # Show unexplained cases
    fig = plt.figure(figsize=[4 * ncols, 4 * nrows], constrained_layout=True)
    for k in range(nrows * ncols):
        case = results.iloc[-(k + 1), :]
        img = sitk.GetArrayFromImage(sitk.ReadImage(dataPath + coco.loadImgs(int(case["image_id"]))[0]['file_name']))
        gt = coco.annToMask(coco.loadAnns(int(case['annotation_id']))[0])
        count += 1
        plt.subplot(nrows, ncols, count)
        plt.imshow(img)
        plt.contour(gt, levels=[0.5], colors=["#ffff00"], linewidths=[2])
        plt.axis("off")
    plt.suptitle("Unexplained cases", size=16)
    plt.savefig("unexplained_cases.png", bbox_inches='tight', pad_inches=0.1)
    plt.show()




