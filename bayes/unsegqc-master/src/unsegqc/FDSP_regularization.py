#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np

def lambda_function(nu):
    return torch.tanh(nu/2)*1/(4*nu+1e-20)

class FDSP_regularization():
    '''
    Class that implements a spatial regularization of a segmentation using a FDSP prior. The model is inferred using 
    variational inference.
    '''

    def __init__(self, config):

        self.dim = config["dim"]
        self.n_iter = config["n_iter"]
        self.config = config
        self.tol = config["tol"]
        self.eps = config["eps"]

    def createConnexityMatrix_2D(self, narrow_band):

        indices = np.arange(self.N_tot)
        indices = np.reshape(indices, self.img_shape)

        indices_nb = np.zeros(self.img_shape)
        indices_nb[narrow_band] = np.arange(self.N)

        ni, nj = self.img_shape
        ii = np.arange(ni)
        jj = np.arange(nj)
        ii, jj = np.meshgrid(ii, jj, indexing='ij')

        # ###########
        # Dimension i

        neighbors_index_i = []
        has_neighbors_i = []

        # We want the n+1 and n+2 neighbors
        for k in range(1, 3):

            # Neighbor to the top
            ii_n = ii - k
            jj_n = jj + 0

            indices_b = indices[ii_n, jj_n]
            indices_nb_b = indices_nb[ii_n, jj_n]

            has_neighbors_b = np.ones(self.img_shape)
            has_neighbors_b[:k, :] = 0

            nb = np.reshape(narrow_band, (-1,))[indices_b].astype(int)
            nb = np.reshape(nb, self.img_shape)
            has_neighbors_b = has_neighbors_b * nb

            indices_nb_b = np.reshape(indices_nb_b, (-1, 1))
            has_neighbors_b = np.reshape(has_neighbors_b, (-1, 1))

            # Neighbor below
            ii_n = ii + k
            ii_n[ii_n >= ni] = ii_n[ii_n >= ni] - ni
            jj_n = jj + 0

            indices_a = indices[ii_n, jj_n]
            indices_nb_a = indices_nb[ii_n, jj_n]

            has_neighbors_a = np.ones(self.img_shape)
            has_neighbors_a[-k:, :] = 0

            nb = np.reshape(narrow_band, (-1,))[indices_a].astype(int)
            nb = np.reshape(nb, self.img_shape)
            has_neighbors_a = has_neighbors_a * nb

            indices_nb_a = np.reshape(indices_nb_a, (-1, 1))
            has_neighbors_a = np.reshape(has_neighbors_a, (-1, 1))

            neighbors_index_i.append(np.concatenate([indices_nb_b, indices_nb_a], axis=1))
            has_neighbors_i.append(np.concatenate([has_neighbors_b, has_neighbors_a], axis=1))

        neighbors_index_i = np.stack(neighbors_index_i)
        has_neighbors_i = np.stack(has_neighbors_i)

        # ###########
        # Dimension j

        neighbors_index_j = []
        has_neighbors_j = []

        # We want the n+1 and n+2 neighbors
        for k in range(1, 3):

            # Neighbor to the left
            ii_n = ii + 0
            jj_n = jj - k

            indices_b = indices[ii_n, jj_n]
            indices_nb_b = indices_nb[ii_n, jj_n]

            has_neighbors_b = np.ones(self.img_shape)
            has_neighbors_b[:, :k] = 0

            nb = np.reshape(narrow_band, (-1,))[indices_b].astype(int)
            nb = np.reshape(nb, self.img_shape)
            has_neighbors_b = has_neighbors_b * nb

            indices_nb_b = np.reshape(indices_nb_b, (-1, 1))
            has_neighbors_b = np.reshape(has_neighbors_b, (-1, 1))

            # Neighbor to the right
            ii_n = ii + 0
            jj_n = jj + k
            jj_n[jj_n >= nj] = jj_n[jj_n >= nj] - nj

            indices_a = indices[ii_n, jj_n]
            indices_nb_a = indices_nb[ii_n, jj_n]

            has_neighbors_a = np.ones(self.img_shape)
            has_neighbors_a[:, -k:] = 0

            nb = np.reshape(narrow_band, (-1,))[indices_a].astype(int)
            nb = np.reshape(nb, self.img_shape)
            has_neighbors_a = has_neighbors_a * nb

            indices_nb_a = np.reshape(indices_nb_a, (-1, 1))
            has_neighbors_a = np.reshape(has_neighbors_a, (-1, 1))

            neighbors_index_j.append(np.concatenate([indices_nb_b, indices_nb_a], axis=1))
            has_neighbors_j.append(np.concatenate([has_neighbors_b, has_neighbors_a], axis=1))

        neighbors_index_j = np.stack(neighbors_index_j)
        has_neighbors_j = np.stack(has_neighbors_j)

        neighbors_index = np.stack([neighbors_index_i, neighbors_index_j]).astype(int)
        has_neighbors = np.stack([has_neighbors_i, has_neighbors_j]).astype(int)

        nb = np.reshape(narrow_band, (-1,))
        neighbors_index = neighbors_index[:,:,nb,:]
        has_neighbors = has_neighbors[:,:,nb,:]

        neighbors_index = torch.from_numpy(neighbors_index)
        has_neighbors = torch.from_numpy(has_neighbors)

        return neighbors_index, has_neighbors

    def createConnexityMatrix_3D(self, narrow_band):

        indices = np.arange(self.N_tot)
        indices = np.reshape(indices, self.img_shape)

        indices_nb = np.zeros(self.img_shape)
        indices_nb[narrow_band] = np.arange(self.N)

        nz, ni, nj = self.img_shape
        zz = np.arange(nz)
        ii = np.arange(ni)
        jj = np.arange(nj)
        zz, ii, jj = np.meshgrid(zz, ii, jj, indexing='ij')

        # ###########
        # Dimension z

        neighbors_index_z = []
        has_neighbors_z = []

        # We want the n+1 and n+2 neighbors
        for k in range(1, 3):

            # Neighbor to the top
            zz_n = zz - k
            ii_n = ii + 0
            jj_n = jj + 0

            indices_b = indices[zz_n, ii_n, jj_n]
            indices_nb_b = indices_nb[zz_n, ii_n, jj_n]

            has_neighbors_b = np.ones(self.img_shape)
            has_neighbors_b[:k, :, :] = 0

            nb = np.reshape(narrow_band, (-1,))[indices_b].astype(int)
            nb = np.reshape(nb, self.img_shape)
            has_neighbors_b = has_neighbors_b * nb

            indices_nb_b = np.reshape(indices_nb_b, (-1, 1))
            has_neighbors_b = np.reshape(has_neighbors_b, (-1, 1))

            # Neighbor below
            zz_n = zz + k
            zz_n[zz_n >= nz] = zz_n[zz_n >= nz] - nz
            ii_n = ii + 0
            jj_n = jj + 0

            indices_a = indices[zz_n, ii_n, jj_n]
            indices_nb_a = indices_nb[zz_n, ii_n, jj_n]

            has_neighbors_a = np.ones(self.img_shape)
            has_neighbors_a[-k:, :, :] = 0

            nb = np.reshape(narrow_band, (-1,))[indices_a].astype(int)
            nb = np.reshape(nb, self.img_shape)
            has_neighbors_a = has_neighbors_a * nb

            indices_nb_a = np.reshape(indices_nb_a, (-1, 1))
            has_neighbors_a = np.reshape(has_neighbors_a, (-1, 1))

            neighbors_index_z.append(np.concatenate([indices_nb_b, indices_nb_a], axis=1))
            has_neighbors_z.append(np.concatenate([has_neighbors_b, has_neighbors_a], axis=1))

        neighbors_index_z = np.stack(neighbors_index_z)
        has_neighbors_z = np.stack(has_neighbors_z)

        # ###########
        # Dimension i

        neighbors_index_i = []
        has_neighbors_i = []

        # We want the n+1 and n+2 neighbors
        for k in range(1, 3):

            # Neighbor to the top
            zz_n = zz + 0
            ii_n = ii - k
            jj_n = jj + 0

            indices_b = indices[zz_n, ii_n, jj_n]
            indices_nb_b = indices_nb[zz_n, ii_n, jj_n]

            has_neighbors_b = np.ones(self.img_shape)
            has_neighbors_b[:, k, :] = 0

            nb = np.reshape(narrow_band, (-1,))[indices_b].astype(int)
            nb = np.reshape(nb, self.img_shape)
            has_neighbors_b = has_neighbors_b * nb

            indices_nb_b = np.reshape(indices_nb_b, (-1, 1))
            has_neighbors_b = np.reshape(has_neighbors_b, (-1, 1))

            # Neighbor below
            zz_n = zz + 0
            ii_n = ii + k
            ii_n[ii_n >= ni] = ii_n[ii_n >= ni] - ni
            jj_n = jj + 0

            indices_a = indices[zz_n, ii_n, jj_n]
            indices_nb_a = indices_nb[zz_n, ii_n, jj_n]

            has_neighbors_a = np.ones(self.img_shape)
            has_neighbors_a[:, -k:, :] = 0

            nb = np.reshape(narrow_band, (-1,))[indices_a].astype(int)
            nb = np.reshape(nb, self.img_shape)
            has_neighbors_a = has_neighbors_a * nb

            indices_nb_a = np.reshape(indices_nb_a, (-1, 1))
            has_neighbors_a = np.reshape(has_neighbors_a, (-1, 1))

            neighbors_index_i.append(np.concatenate([indices_nb_b, indices_nb_a], axis=1))
            has_neighbors_i.append(np.concatenate([has_neighbors_b, has_neighbors_a], axis=1))

        neighbors_index_i = np.stack(neighbors_index_i)
        has_neighbors_i = np.stack(has_neighbors_i)

        # ###########
        # Dimension j

        neighbors_index_j = []
        has_neighbors_j = []

        # We want the n+1 and n+2 neighbors
        for k in range(1, 3):

            # Neighbor to the left
            zz_n = zz + 0
            ii_n = ii + 0
            jj_n = jj - k

            indices_b = indices[zz_n, ii_n, jj_n]
            indices_nb_b = indices_nb[zz_n, ii_n, jj_n]

            has_neighbors_b = np.ones(self.img_shape)
            has_neighbors_b[:, :, :k] = 0

            nb = np.reshape(narrow_band, (-1,))[indices_b].astype(int)
            nb = np.reshape(nb, self.img_shape)
            has_neighbors_b = has_neighbors_b * nb

            indices_nb_b = np.reshape(indices_nb_b, (-1, 1))
            has_neighbors_b = np.reshape(has_neighbors_b, (-1, 1))

            # Neighbor to the right
            zz_n = zz + 0
            ii_n = ii + 0
            jj_n = jj + k
            jj_n[jj_n >= nj] = jj_n[jj_n >= nj] - nj

            indices_a = indices[zz_n, ii_n, jj_n]
            indices_nb_a = indices_nb[zz_n, ii_n, jj_n]

            has_neighbors_a = np.ones(self.img_shape)
            has_neighbors_a[:, :, -k:] = 0

            nb = np.reshape(narrow_band, (-1,))[indices_a].astype(int)
            nb = np.reshape(nb, self.img_shape)
            has_neighbors_a = has_neighbors_a * nb

            indices_nb_a = np.reshape(indices_nb_a, (-1, 1))
            has_neighbors_a = np.reshape(has_neighbors_a, (-1, 1))

            neighbors_index_j.append(np.concatenate([indices_nb_b, indices_nb_a], axis=1))
            has_neighbors_j.append(np.concatenate([has_neighbors_b, has_neighbors_a], axis=1))

        neighbors_index_j = np.stack(neighbors_index_j)
        has_neighbors_j = np.stack(has_neighbors_j)

        neighbors_index = np.stack([neighbors_index_z, neighbors_index_i, neighbors_index_j]).astype(int)
        has_neighbors = np.stack([has_neighbors_z, has_neighbors_i, has_neighbors_j]).astype(int)

        nb = np.reshape(narrow_band, (-1,))
        neighbors_index = neighbors_index[:, :, nb, :]
        has_neighbors = has_neighbors[:, :, nb, :]

        neighbors_index = torch.from_numpy(neighbors_index)
        has_neighbors = torch.from_numpy(has_neighbors)

        return neighbors_index, has_neighbors

    def initialization(self, appearance_ratio):
        '''
        :param appearance_ratio: N
        :return: alpha, mu_w, xi, rn
        '''

        alpha = torch.ones(1, dtype=float) * self.config["alpha0"]
        xi = torch.randn(self.N, dtype=float) ** 2
        rn = torch.cat([appearance_ratio.view(-1,1), 1 - appearance_ratio.view(-1,1)], dim=1)
        w = rn[:,0]*10 - rn[:,1]*10
        sigma_w = torch.randn(self.N, dtype=float) ** 2 + self.eps

        return alpha, w, sigma_w, xi, rn

    def q_z(self, w, w_squared, appearance_ratio, lambda_xi, expit_xi, xi):
        '''
        :param w: N
        :param w_squared: N
        :param appearance_ratio: N
        :param lambda_xi: N
        :param expit_xi: N
        :param xi: N
        :return: rn
        '''

        targets_expit_xi = appearance_ratio * expit_xi  # N

        b = lambda_xi * (w_squared - xi ** 2)

        rho_1 = targets_expit_xi * torch.exp(1 / 2 * (w - xi) - b)
        rho_1 = rho_1.view(-1, 1)

        rho_2 = (expit_xi - targets_expit_xi) * torch.exp(- 1 / 2 * (w + xi) - b)
        rho_2 = rho_2.view(-1, 1)

        rn = torch.cat([rho_1, rho_2], dim=1)
        rn_sum = torch.sum(rn, dim=1)
        rn = torch.einsum('nk,n->nk', rn, 1 / rn_sum)

        return rn

    def q_w(self, w, sigma_w, neighbors_index, has_neighbors, lambda_xi, rn, alpha):
        '''
        :param w: N
        :param sigma_w: N
        :param neighbors_index: dim x neighbor order x N x (before - after) (Index of neighbors)
        :param has_neighbors: dim x neighbor order x N x (before - after)
        :param lambda_xi: N
        :param rn: N x 2
        :param alpha: scalar
        :return: mu_w, sigma_w, bf_mu_w, trace_bf_bf_sigma_w
        '''

        update = np.random.choice(np.arange(self.N), int(self.config["prop"]*self.N), replace=False)

        reg_part = 1/2 * torch.einsum('dnk->n', has_neighbors[:,1,:,:]) * alpha # N
        sigma_w_new = 2*lambda_xi + reg_part
        sigma_w_new = 1/sigma_w_new # N
        sigma_w[update] = sigma_w_new[update]

        w_neighbors = 0
        for d in range(self.dim):
            w_neighbors += torch.einsum('n,n->n', has_neighbors[d,1,:,0], w[neighbors_index[d,1,:,0]]) +\
                               torch.einsum('n,n->n', has_neighbors[d,1,:,1], w[neighbors_index[d,1,:,1]]) # N
        w_neighbors = 1/2 * w_neighbors * alpha # N

        w_new = torch.einsum('n,n->n', sigma_w, (rn[:,0] - 1/2) + w_neighbors)  # N
        w[update] = w_new[update]

        w_squared = sigma_w + w**2

        return w, sigma_w, w_squared

    def q_alpha(self, w, sigma_w, neighbors_index, has_neighbors):
        '''
        :param w: N
        :param sigma_w: N
        :param neighbors_index: dim x neighbor order x N x (before - after) (Index of neighbors)
        :param has_neighbors: dim x neighbor order x N x (before - after)
        :return: alpha
        '''

        E = 0
        for d in range(self.dim):

            in_sum = (has_neighbors[d, 0, :, 0] > 0) & (has_neighbors[d, 0, :, 1] > 0)
            w_b = w[neighbors_index[d, 0, :, 0]]
            cov_b = sigma_w[neighbors_index[d, 0, :, 0]]
            w_a = w[neighbors_index[d, 0, :, 1]]
            cov_a = sigma_w[neighbors_index[d, 0, :, 1]]
            E += torch.sum(((w_a - w_b)** 2 + cov_a + cov_b)[in_sum])

        alpha = 1 / (2 * self.N) * E
        alpha = 1 / alpha

        return alpha

    def q_xi(self, w_squared):
        '''
        :param w_squared: N
        :return: xi, lambda_xi, expit_xi
        '''

        xi = torch.sqrt(w_squared) # N

        # Compute values that will be used later
        lambda_xi = lambda_function(xi)
        expit_xi = torch.sigmoid(xi)

        return xi, lambda_xi, expit_xi


    def lower_bound(self, appearance_ratio, rn, lambda_xi, expit_xi, xi, w, w_squared, sigma_w, alpha):
        '''
        :param appearance_ratio: N
        :param rn: N x 2
        :param lambda_xi: N
        :param expit_xi: N
        :param xi: N
        :param w: N
        :param w_squared: N
        :param sigma_w: L x L
        :param alpha: scalar
        :return: lower bound
        '''

        p_I = rn * torch.log(torch.cat([appearance_ratio.view((-1,1)), (1 - appearance_ratio).view((-1,1))], dim=1))
        p_I = torch.sum(p_I)

        b = lambda_xi * (w_squared - xi ** 2)  # N
        rho_1 = torch.log(expit_xi) + 1 / 2 * (w - xi) - b
        rho_1 = rn[:, 0].view((-1, 1)) * rho_1.view((-1, 1))
        rho_2 = torch.log(expit_xi) - 1 / 2 * (w + xi) - b
        rho_2 = rn[:, 1].view((-1, 1)) * rho_2.view((-1, 1))
        p_Z = torch.sum(rho_1 + rho_2)

        p_W = self.N/2 * torch.log(alpha)

        q_Z = rn * torch.log(rn)
        q_Z[torch.isnan(q_Z)] = 0  # xlogx = 0 for x -> 0
        q_Z = torch.sum(q_Z)

        q_W = - 1 / 2 * torch.log(sigma_w)
        q_W = torch.sum(q_W)

        lb = p_I + p_Z + p_W - q_Z - q_W

        return lb

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

        # Create a matrix that stores the index of the neighbors for each voxel in all dimensions
        if self.dim == 2:
            neighbors_index, has_neighbors = self.createConnexityMatrix_2D(narrow_band)
        else:
            neighbors_index, has_neighbors = self.createConnexityMatrix_3D(narrow_band)

        # Initialization
        appearance_ratio = torch.from_numpy(appearance_ratio)
        alpha, w, sigma_w, xi, rn = self.initialization(appearance_ratio)
        lambda_xi = lambda_function(xi)

        i = 0
        converged = 0
        lower_bound_list = []

        while i < self.n_iter:

            w, sigma_w, w_squared = self.q_w(w, sigma_w, neighbors_index, has_neighbors, lambda_xi, rn, alpha)

            alpha = self.q_alpha(w, sigma_w, neighbors_index, has_neighbors)

            xi, lambda_xi, expit_xi = self.q_xi(w_squared)

            rn = self.q_z(w, w_squared, appearance_ratio, lambda_xi, expit_xi, xi)

            lb = self.lower_bound(appearance_ratio, rn, lambda_xi, expit_xi, xi, w, w_squared, sigma_w, alpha)

            if i > 1:
                # We check that the lower bounds is increasing
                assert lower_bound_list[-1] <= lb

            lower_bound_list.append(lb)

            if i > 1:
                diff = lower_bound_list[-1] - lower_bound_list[-2]
                if torch.abs(diff / lower_bound_list[-2]) < self.tol:
                    i = self.n_iter
                    converged = 1

            i += 1

        # Check convergence
        if not converged and self.n_iter > 0:
            self.converged = False
        else:
            self.converged = True

        self.lower_bound_list = [float(i.cpu()) for i in lower_bound_list]
        self.rn = rn.cpu().numpy()
        self.mu_w = w.cpu().numpy()
        self.xi = xi.cpu().numpy()
        self.sigma_w = sigma_w.cpu().numpy()
        self.alpha = alpha.cpu().numpy()
        self.neighbors_index = neighbors_index.cpu().numpy()
        self.has_neighbors = has_neighbors.cpu().numpy()

        self.fitted = True

        return self


if __name__ =="__main__":

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from sklearn.mixture import BayesianGaussianMixture
    from scipy.special import expit

    # Create an image
    img = np.zeros((100,100))
    # Square
    img[40:60,40:60] = 1
    # Circle
    # ii, jj = np.arange(100), np.arange(100)
    # ii, jj = np.meshgrid(ii, jj, indexing='ij')
    # ii, jj = np.reshape(ii, (-1,)), np.reshape(jj, (-1,))
    # index = ((ii-50)**2 + (jj-50)**2) <= 20**2
    # img = np.reshape(img, (-1,))
    # img[index] = 1
    # img = np.reshape(img, (100,100))
    narrow_band = np.zeros(img.shape, dtype=np.bool)
    narrow_band[10:-10,10:-10] = 1

    img = img + np.random.random(img.shape) * (0.5 - 0) + 0
    img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))

    # # Plot the image
    # plt.figure(figsize=(3,3))
    # plt.imshow(img, cmap="gray")
    # plt.axis("off")
    # plt.tight_layout()
    # plt.show()

    # Model
    config = {
        "dim": 2,
        "tol": 1e-8,
        "eps": 1e-8,
        "n_iter": 200,
        "alpha0": 1,
        "use_gpu": False,
        "prop": 1
    }
    model = FDSP_regularization(config)

    # Appearance model
    appearance_model = BayesianGaussianMixture(n_components=2)
    appearance_model.fit(np.reshape(img[narrow_band], (-1,1)))
    X = appearance_model.predict_proba(np.reshape(img[narrow_band], (-1,1)))[:,0]
    X[X == 0] = 1e-3
    X[X == 1] = 1-1e-3

    # Inputs
    X = np.reshape(X, (-1,))

    model.fit(X, narrow_band)

    print("Alpha's value after convergence:", model.alpha)

    # Plot the lower bound
    plt.figure(figsize=[3,3])
    plt.plot(model.lower_bound_list)
    plt.xlabel("Iterations")
    plt.ylabel("Lower bound")
    plt.show()

    # Plot the result
    rn = model.rn[:,0]
    mu_w = np.ones(img.shape) * float(X[0] > 0.5)
    mu_w[narrow_band] = expit(model.mu_w)
    X_full = np.ones(img.shape) * float(X[0] > 0.5)
    X_full[narrow_band] = X

    plt.figure(figsize=[3,3])
    plt.imshow(img, cmap="gray")
    handles= []
    # plt.contour(np.reshape(rn, img.shape), levels=[0.5], colors='red', linewidths=2)
    handles.append(Line2D([], [], color='red', linestyle="-", label='P(I|Z)', linewidth=2))
    plt.contour(mu_w, levels=[0.5], colors='green', linewidths=2)
    handles.append(Line2D([], [], color='green', linestyle="-", label='P(Z|W)', linewidth=2))
    # plt.contour(np.reshape(X_full, img.shape), levels=[0.5], colors='blue', linewidths=2)
    handles.append(Line2D([], [], color='blue', linestyle="-", label='P(Z|I,W)', linewidth=2))
    plt.legend(handles=handles)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
