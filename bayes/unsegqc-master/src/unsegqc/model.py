#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scipy.ndimage.morphology import distance_transform_edt
import numpy as np
import sys
sys.path.append("/home/baudelan/SegmentationQualityControl/VariationalSMM/")
from unsegqc.bayesian_smm import SMM
from unsegqc.GLSP_regularization import GLSP_regularization
from unsegqc.FDSP_regularization import FDSP_regularization
from unsegqc.MRF_regularization import MRF_regularization
from medpy.metric.binary import dc
from scipy.special import expit

class ProbModel():

    def __init__(self, config):

        self.config = config
        self.dim = config["dim"]
        assert self.dim == 2 or self.dim == 3
        self.regularization_method = config["regularization_method"]
        if self.regularization_method not in ["GLSP", "FDSP", "MRF"]:
            raise ValueError("Regularization method not recognized")

    def create_narrow_band(self, seg):
        '''
        :param img: img_shape
        :param spacing: tuple
        :return: narrow_band
        '''

        dm1 = distance_transform_edt(seg) / self.config["bound"]
        dm1[dm1 >= 1] = 0
        dm1[dm1 > 0] = 1

        dm2 = distance_transform_edt(1-seg) / self.config["bound"]
        dm2[dm2 >= 1] = 0
        dm2[dm2 > 0] = 1

        narrow_band = np.logical_or(dm1, dm2) # img_shape
        narrow_band = narrow_band.astype(bool)

        return narrow_band

    def crop(self, seg, imgs, narrow_band):
        '''
        :param seg: img_shape
        :param: imgs: P x img_shape
        :param: narrow_band: img_shape
        :return: seg cropped, imgs cropped, narrow_band cropped
        '''

        margin = self.config["margin"]

        if self.dim == 3:
            zmax, zmin, imax, imin, jmax, jmin = -np.PINF, np.PINF, -np.PINF, np.PINF, -np.PINF, np.PINF
            zz, ii, jj = np.where(seg)
            zmax, imax, jmax = max(zmax, max(zz)), max(imax, max(ii)), max(jmax, max(jj))
            zmin, imin, jmin = min(zmin, min(zz)), min(imin, min(ii)), min(jmin, min(jj))

            self.offset = np.array([[max(zmin - margin, 0), min(seg.shape[0], zmax + margin)],
                                    [max(imin - margin, 0), min(seg.shape[1], imax + margin)],
                                    [max(jmin - margin, 0), min(seg.shape[2], jmax + margin)]])

            seg_crop = seg[max(zmin - margin, 0):min(seg.shape[0], zmax + margin),
                           max(imin - margin, 0):min(seg.shape[1], imax + margin),
                           max(jmin - margin, 0):min(seg.shape[2], jmax + margin)]
            narrow_band = narrow_band[max(zmin - margin, 0):min(narrow_band.shape[0], zmax + margin),
                                      max(imin - margin, 0):min(narrow_band.shape[1], imax + margin),
                                      max(jmin - margin, 0):min(narrow_band.shape[2], jmax + margin)]
            imgs_crop = imgs[:,
                             max(zmin - margin, 0):min(imgs.shape[1], zmax + margin),
                             max(imin - margin, 0):min(imgs.shape[2], imax + margin),
                             max(jmin - margin, 0):min(imgs.shape[3], jmax + margin)]

        elif self.dim == 2:
            imax, imin, jmax, jmin = -np.PINF, np.PINF, -np.PINF, np.PINF
            ii, jj = np.where(seg)
            imax, jmax = max(imax, max(ii)), max(jmax, max(jj))
            imin, jmin = min(imin, min(ii)), min(jmin, min(jj))

            self.offset = np.array([[max(imin - margin, 0), min(seg.shape[0], imax + margin)],
                                    [max(jmin - margin, 0), min(seg.shape[1], jmax + margin)]])

            seg_crop = seg[max(imin - margin, 0):min(seg.shape[0], imax + margin),
                           max(jmin - margin, 0):min(seg.shape[1], jmax + margin)]
            narrow_band = narrow_band[max(imin - margin, 0):min(narrow_band.shape[0], imax + margin),
                                      max(jmin - margin, 0):min(narrow_band.shape[1], jmax + margin)]
            imgs_crop = imgs[:,
                             max(imin - margin, 0):min(imgs.shape[1], imax + margin),
                             max(jmin - margin, 0):min(imgs.shape[2], jmax + margin)]



        else:
            raise ValueError("Dimension not recognized")

        return seg_crop, imgs_crop, narrow_band

    def compute_appearance_ratio(self, imgs, seg, narrow_band):
        '''
        :param imgs: P x img_shape (P is the number of modalities)
        :param seg: img_shape
        :param narrow_band: img_shape
        :return: appearance_ratio
        '''

        # Foreground and background according to the input segmentation
        mask_foreground = np.logical_and(narrow_band, seg.astype(bool))
        FG = imgs[:, mask_foreground]  # P x Nforeground
        mask_background = np.logical_and(narrow_band, np.invert(seg.astype(bool))) # P x Nbackground
        BG = imgs[:, mask_background]

        n_comp = self.config["n_comp"]
        tol = self.config["SMM_tol"]
        n_iter = self.config["SMM_n_iter"]
        min_covar = self.config["SMM_min_covar"]

        # Fit a student mixture model for the foreground
        FG = np.transpose(FG, (1,0)) # N x P
        flag = False
        while not flag and tol < 10:
            mix_t_foreground = SMM(n_components=n_comp,
                                   random_state=0,
                                   tol=tol,
                                   min_covar=min_covar,
                                   n_iter=n_iter,
                                   n_init=1)
            mix_t_foreground.fit(FG)
            flag = mix_t_foreground.fitted
            tol = tol * 10
        if not flag:
            raise Exception("Fitting a t-mixture model for the foreground failed")

        # Fit a student mixture model for the background
        BG = np.transpose(BG, (1,0)) # P x N
        flag = False
        while not flag and tol < 10:
            mix_t_background = SMM(n_components=n_comp,
                                   random_state=0,
                                   tol=tol,
                                   min_covar=min_covar,
                                   n_iter=n_iter,
                                   n_init=1)
            mix_t_background.fit(BG)
            flag = mix_t_background.fitted
            tol = tol * 10
        if not flag:
            raise Exception("Fitting a t-mixture model for the background failed")

        # Compute p(I|Z)
        imgs = imgs[:,narrow_band]
        imgs = np.transpose(imgs, (1,0)) # N x P
        prob_b = np.sum(mix_t_background.predict_mixture_density(imgs), axis=1)
        prob_f = np.sum(mix_t_foreground.predict_mixture_density(imgs), axis=1)

        appearance_ratio = prob_f / (prob_b + prob_f)

        self.mix_t_foreground = mix_t_foreground
        self.mix_t_background = mix_t_background

        return appearance_ratio

    def find_contours(self, prob_map):
        '''
        :param prob_map: img_shape
        :return: contours (voxels belonging to the isoline 0.5)
        '''

        tmp = np.ones(prob_map.shape)
        tmp[np.invert(prob_map.mask)] = prob_map[np.invert(prob_map.mask)]
        tmp[tmp >= 0.5] = 1
        tmp[tmp < 1] = 0
        dm = distance_transform_edt(tmp)
        dm[dm > 1] = 0
        dm[dm > 0] = 1
        dm[prob_map.mask] = 0

        contours = np.ma.masked_where(dm == 0, dm)

        return contours

    def compute_signed_distance(self, seg_contours, posterior, spacing):
        '''
        Compute the distance for voxels belonging to the isoline 0.5 in the input segmentation
        to the isoline 0.5 of the posterior.

        :param seg_contours: img_shape
        :param posterior: img_shape
        :param spacing: tuple
        :return: signed distance
        '''

        # We check that the posterior has foreground and background
        unique_label = posterior[self.narrow_band]
        unique_label[unique_label >= 0.5] = 1
        unique_label[unique_label < 1] = 0
        unique_label = np.unique(unique_label)

        if len(unique_label) > 1:

            # Distance of the foreground voxels to the isoline (counted as negative distance)
            posterior_bin = np.ones(posterior.shape, dtype=np.float)
            posterior_bin[self.narrow_band] = posterior[self.narrow_band]
            posterior_bin[posterior_bin >= 0.5] = 1
            posterior_bin[posterior_bin < 1] = 0
            posterior_bin[np.invert(self.narrow_band)] = 1 # To be ignored when computing the distance
            dm1 = distance_transform_edt(posterior_bin, sampling=spacing)

            # Distance of the background voxels to the isoline (counted as positive distance)
            posterior_bin = np.ones(posterior.shape, dtype=np.float)
            posterior_bin[self.narrow_band] = posterior[self.narrow_band]
            posterior_bin[posterior_bin >= 0.5] = 1
            posterior_bin[posterior_bin < 1] = 0
            posterior_bin = np.abs(1 - posterior_bin)
            posterior_bin[np.invert(self.narrow_band)] = 1  # To be ignored when computing the distance
            dm2 = distance_transform_edt(posterior_bin, sampling=spacing)

            signed_dist = dm2 - dm1

        elif int(unique_label) == 1:

            # There is only foreground
            signed_dist = -np.ones(posterior.shape) * np.PINF

        else:

            # There is only background
            signed_dist = np.ones(posterior.shape) * np.PINF

        signed_dist = np.ma.masked_where(seg_contours.mask, signed_dist)

        return signed_dist

    def compute_dice(self, seg, posterior):
        '''
        :param seg: img_shape
        :param posterior: img_shape
        :return: dice score
        '''

        bin_post = posterior[self.narrow_band].data
        bin_post[bin_post >= 0.5] = 1
        bin_post[bin_post < 1] = 0

        bin_seg = seg[self.narrow_band]
        bin_seg[bin_seg >= 0.5] = 1
        bin_seg[bin_seg < 1] = 0

        if not np.sum(bin_seg) + np.sum(bin_post):
            return 1.0
        else:
            return dc(bin_seg, bin_post)

        return dice

    def assess_seg_quality(self, seg, imgs, spacing, spatial_prior_params=None, mask=None):
        '''
        :param seg: img_shape
        :param imgs: P x img_shape (P is the number of image modalities)
        :param spacing: tuple
        :param spatial_prior_params: list
        :param mask: img_shape (if we want to mask a specific part of the image)
        '''

        assert np.array_equal(np.unique(seg), [0,1])

        img_shape = seg.shape

        # By default there is no narrow band
        if type(mask) == type(None):
            narrow_band = np.ones(img_shape, dtype=np.bool)
        else:
            narrow_band = mask.astype(np.bool)

        # First we crop the image around the structure of interest
        seg, imgs, narrow_band = self.crop(seg, imgs, narrow_band)
        self.seg = seg
        self.img_shape = seg.shape
        self.imgs = imgs

        # We work on a narrow band along the segmentation boundary
        if self.config["use_narrow_band"]:
            nb = self.create_narrow_band(seg)  # img_shape
            narrow_band = np.logical_and(nb, narrow_band)
        self.narrow_band = narrow_band

        # We want to assess the quality of the segmentation using a probabilistic model that builds a new segmentation
        # that will be used as a comparison tool
        self.probabilistic_quality_control(imgs, seg, narrow_band, spacing)

    def probabilistic_quality_control(self, imgs, seg, narrow_band, spacing):
        '''
        :param imgs: P x img_shape (P is the number of modalities)
        :param seg: img_shape
        :param narrow_band: img_shape
        :param spacing: tuple
        '''

        # We first fit two Student-t mixtures modelling the appearance of the foreground and the background
        print("Computing appearance ratio")
        appearance_ratio = self.compute_appearance_ratio(imgs, seg, narrow_band) # N

        self.appearance_ratio = np.zeros(self.img_shape)
        self.appearance_ratio[self.narrow_band] = appearance_ratio
        self.appearance_ratio = np.ma.masked_where(np.invert(self.narrow_band), self.appearance_ratio)

        # Then we apply a spatial regularization
        print("Fitting the regularization model")
        if self.regularization_method == "GLSP":
            self.model = GLSP_regularization(self.config)
        elif self.regularization_method == "MRF":
            self.model = MRF_regularization(self.config)
        else:
            self.model = FDSP_regularization(self.config)
        self.model.fit(appearance_ratio, narrow_band)

        # Compute the label posterior
        posterior = np.zeros(self.img_shape)
        posterior[self.narrow_band] = self.model.rn[:,0]
        self.posterior = np.ma.masked_where(np.invert(self.narrow_band), posterior)

        # Compute the prior when the regularization is "GLSP" or "FDSP"
        if self.regularization_method != "MRF":
            prior = np.zeros(self.img_shape)
            if self.regularization_method == "GLSP":
                prior[self.narrow_band] = expit(self.model.bf_mu_w)
                prior = np.ma.masked_where(np.invert(self.narrow_band), prior)
                self.prior = prior
            else:
                prior[self.narrow_band] = expit(self.model.mu_w)
            self.prior = np.ma.masked_where(np.invert(self.narrow_band), prior)


        # Compute the signed asymetric surface error and the Dice score
        seg_contours = self.find_contours(np.ma.masked_where(np.invert(narrow_band), seg))
        self.signed_dist = self.compute_signed_distance(seg_contours, self.posterior, spacing)
        self.ase = np.mean(np.abs(self.signed_dist[np.invert(self.signed_dist.mask)]))
        self.dice_score = self.compute_dice(seg, self.posterior)

        # Coordinates of points belonging to the input segmentation border
        # with their absolute distance to the model
        signed_dist_mask = np.invert(self.signed_dist.mask).astype(int)
        index = np.where(signed_dist_mask == 1)
        abs_dist_on_border = list(index) + [np.abs(self.signed_dist.data[np.invert(self.signed_dist.mask)])]
        abs_dist_on_border = np.concatenate([np.reshape(x, (-1,1)) for x in abs_dist_on_border], axis=1)
        self.abs_dist_on_border = abs_dist_on_border









