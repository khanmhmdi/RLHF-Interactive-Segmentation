#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
from sklearn.neighbors import radius_neighbors_graph



class MRF_regularization():
    '''
    Class that implements a spatial regularization of a segmentation using a Markov random field prior. The model is
    inferred using variational inference.
    '''

    def __init__(self, config):

        self.dim = config["dim"]
        self.n_iter = config["n_iter"]
        self.config = config
        self.tol = config["tol"]
        self.eps = config["eps"]

    def initialization(self, appearance_ratio, narrow_band):
        '''
        :param appearance_ratio: N
        :param narrow_band: img_shape
        :return: connectivity_matrix, rn
        '''

        rn = torch.cat([appearance_ratio.view(-1,1), 1 - appearance_ratio.view(-1,1)], dim=1)

        if self.dim == 3:

            oz = np.arange(0, self.img_shape[0])
            oi = np.arange(0, self.img_shape[1])
            oj = np.arange(0, self.img_shape[2])

            zz, ii, jj = np.meshgrid(oz, oi, oj, indexing="ij")
            zz = zz[narrow_band]
            ii = ii[narrow_band]
            jj = jj[narrow_band]

            coord = np.concatenate([np.reshape(zz, (-1, 1)),
                                    np.reshape(ii, (-1, 1)),
                                    np.reshape(jj, (-1, 1))], axis=1)
            connectivity_matrix = radius_neighbors_graph(coord,
                                                         self.config["connectivity"],
                                                         mode='connectivity',
                                                         include_self=False,
                                                         n_jobs=-1)

        else:

            oi = np.arange(0, self.img_shape[0])
            oj = np.arange(0, self.img_shape[1])

            ii, jj = np.meshgrid(oi, oj, indexing="ij")
            ii = ii[narrow_band]
            jj = jj[narrow_band]

            coord = np.concatenate([np.reshape(ii, (-1, 1)),
                                    np.reshape(jj, (-1, 1))], axis=1)
            connectivity_matrix = radius_neighbors_graph(coord,
                                                         self.config["connectivity"],
                                                         mode='connectivity',
                                                         include_self=False,
                                                         n_jobs=-1)

        connectivity_matrix = connectivity_matrix.tocoo()
        index = np.concatenate([np.reshape(connectivity_matrix.row, (1, -1)),
                                np.reshape(connectivity_matrix.col, (1, -1))], axis=0).astype(int)
        connectivity_matrix = torch.sparse.FloatTensor(torch.from_numpy(index),
                                                       torch.from_numpy(connectivity_matrix.data),
                                                       torch.Size([self.N, self.N]))

        beta = self.config["beta"]

        return connectivity_matrix, rn, beta

    def q_z(self, rn, appearance_ratio, connectivity_matrix, beta):
        '''
        :param rn: N x 2
        :param appearance_ratio: N
        :param connectivity_matrix: N x N
        :param beta: scalar
        :return: rn
        '''

        vc = torch.sparse.mm(connectivity_matrix, rn)  # N x 2
        vc = beta * vc

        appearance_ratio = torch.cat([appearance_ratio.view((-1,1)), (1-appearance_ratio).view((-1,1))], dim=1)
        rn = torch.log(appearance_ratio) + vc # N x 2

        rn = torch.softmax(rn, dim=1)

        return  rn

    def fit(self, appearance_ratio, narrow_band):
        '''
        :param appearance_ratio: N
        :param: narrow_band: img_shape
        '''

        appearance_ratio = appearance_ratio.astype(float)
        appearance_ratio[appearance_ratio == 0] = self.eps
        appearance_ratio[appearance_ratio == 1] = 1 - self.eps
        self.N = appearance_ratio.shape[0]
        self.img_shape = narrow_band.shape
        self.N_tot = np.prod(self.img_shape)

        # Initialization
        appearance_ratio = torch.from_numpy(appearance_ratio)
        connectivity_matrix, rn, beta = self.initialization(appearance_ratio, narrow_band)

        i = 0
        converged = 0

        while i < self.n_iter:

            rn_old = torch.empty_like(rn, dtype=float)
            rn_old.data = rn.clone()
            rn = self.q_z(rn, appearance_ratio, connectivity_matrix, beta)

            if i > 1:
                if torch.allclose(rn, rn_old, rtol=self.tol):
                    i = self.n_iter
                    converged = 1

            i += 1

        # Check convergence
        if not converged and self.n_iter > 0:
            self.converged = False
        else:
            self.converged = True

        self.rn = rn.cpu().numpy() # N x 2
        self.fitted = True

        return self


if __name__ =="__main__":

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from sklearn.mixture import BayesianGaussianMixture

    # Create an image
    img = np.zeros((100,100))
    img[40:60,40:60] = 1
    img = img + np.random.random(img.shape) * (0.5 - 0) + 0
    img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))

    # Plot the image
    plt.figure(figsize=(3,3))
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    # Model
    config = {
        "dim": 2,
        "tol": 1e-3,
        "eps": 1e-8,
        "n_iter": 100,
        "beta": 1,
        "connectivity": 1,
    }
    model = MRF_regularization(config)

    # Appearance model
    appearance_model = BayesianGaussianMixture(n_components=2)
    appearance_model.fit(np.reshape(img, (-1,1)))
    X = appearance_model.predict_proba(np.reshape(img, (-1,1)))[:,0]

    # Inputs
    narrow_band = np.ones(img.shape, dtype=np.bool)
    X = np.reshape(X, (-1,))

    model.fit(X, narrow_band)

    # Plot the result
    rn = model.rn[:,0]
    plt.figure(figsize=[3,3])
    plt.imshow(img, cmap="gray")
    handles= []
    plt.contour(np.reshape(rn, img.shape), levels=[0.5], colors='red', linewidths=2)
    handles.append(Line2D([], [], color='red', linestyle="-", label='P(I|Z)', linewidth=2))
    plt.contour(np.reshape(X, img.shape), levels=[0.5], colors='blue', linewidths=2)
    handles.append(Line2D([], [], color='blue', linestyle="-", label='P(Z|I,W)', linewidth=2))
    plt.legend(handles=handles)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
