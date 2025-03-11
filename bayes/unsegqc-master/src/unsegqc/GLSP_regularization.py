#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
import scipy.sparse


def multivariate_normal_pdf_2D(scale):

    ii = np.arange(-scale[0] - 1, scale[0] + 2, 1)
    jj = np.arange(-scale[1] - 1, scale[1] + 2, 1)

    ni, nj = len(ii), len(jj)

    ii, jj = np.meshgrid(ii, jj, indexing='ij')

    ii = np.reshape(ii, (-1, 1))
    jj = np.reshape(jj, (-1, 1))

    ivar = (scale[0] / 3) ** 2
    jvar = (scale[1] / 3) ** 2
    res = (ii ** 2 / scale[0]**2 + jj ** 2 / scale[1]**2).astype(np.float64)
    ind_0 = np.where(res > 1)  # Set to 0 if > to 3 standard deviation
    ind_1 = np.where(res <= 1)
    res[ind_0] = 0
    res[ind_1] = 1 / (2 * np.pi) * 1 / (ivar * jvar)**0.5 * np.exp(-1 / 2 * (1/ivar * ii[ind_1]**2 + 1/jvar * jj[ind_1]**2))

    res = np.reshape(res, (ni, nj)).astype(float)

    return res

def multivariate_normal_pdf_3D(scale):

    zz = np.arange(-scale[0] - 1, scale[0] + 2, 1)
    ii = np.arange(-scale[1] - 1, scale[1] + 2, 1)
    jj = np.arange(-scale[2] - 1, scale[2] + 2, 1)

    nz, ni, nj = len(zz), len(ii), len(jj)

    zz, ii, jj = np.meshgrid(zz, ii, jj, indexing='ij')

    zz = np.reshape(zz, (-1, 1))
    ii = np.reshape(ii, (-1, 1))
    jj = np.reshape(jj, (-1, 1))

    zvar = (scale[0] / 3) ** 2
    ivar = (scale[1] / 3) ** 2
    jvar = (scale[2] / 3) ** 2
    res = (zz ** 2 / scale[0]**2 + ii ** 2 / scale[1]**2 + jj ** 2 / scale[2]**2).astype(np.float64)
    ind_0 = np.where(res > 1)  # Set to 0 if > to 3 standard deviation
    ind_1 = np.where(res <= 1)
    res[ind_0] = 0
    res[ind_1] = 1 / (2 * np.pi) * 1 / (zvar * ivar * jvar)**0.5 * np.exp(-1 / 2 * (1/zvar * zz[ind_1]**2 + 1/ivar * ii[ind_1]**2 + 1/jvar * jj[ind_1]**2))

    res = np.reshape(res, (nz, ni, nj)).astype(float)

    return res

class GLSP_regularization():
    '''
    Class that implements a spatial regularization of a segmentation using a GLSP prior. The model is inferred using 
    a type II maximum likelihood approach with a Laplace approximation.
    '''

    def __init__(self, config):

        self.dim = config["dim"]
        self.n_iter = config["n_iter"]
        self.nr_n_iter = config["nr_n_iter"]
        self.config = config
        self.tol = config["tol"]
        self.nr_tol = config["nr_tol"]
        self.eps = config["eps"]

    def createBasisFunctions_2D(self, narrow_band):

        flatten_indices = np.arange(self.N_tot)
        flatten_indices = np.reshape(flatten_indices, self.img_shape)

        basis_functions = []
        icoord = []
        jcoord = []

        scale = [self.config["spatial_prior_params"]["iscale"], self.config["spatial_prior_params"]["jscale"]]
        istep, jstep = self.config["spatial_prior_params"]["istep"], self.config["spatial_prior_params"]["jstep"]
        layout = self.config["spatial_prior_params"]["layout"]
        assert (layout == "square" or layout == "quinconce")

        ii_basis_centers = np.arange(self.img_shape[0])
        jj_basis_centers = np.arange(self.img_shape[1])
        ii_basis_centers = ii_basis_centers % istep
        ii_basis_centers = np.where(ii_basis_centers == 0)[0]
        jj_basis_centers = jj_basis_centers % jstep
        jj_basis_centers = np.where(jj_basis_centers == 0)[0]

        if layout == "quinconce":
            ii_basis_centers = ii_basis_centers[0::2]
            jj_basis_centers = jj_basis_centers[0::2]

        ii_basis_centers, jj_basis_centers = np.meshgrid(ii_basis_centers, jj_basis_centers, indexing="ij")
        ii_basis_centers, jj_basis_centers = np.reshape(ii_basis_centers, (-1, 1)), np.reshape(jj_basis_centers,
                                                                                               (-1, 1))

        if layout == "quinconce":
            ii_sup = (ii_basis_centers + istep).astype(np.int)
            jj_sup = (jj_basis_centers + jstep).astype(np.int)
            ind_to_remove = np.logical_or(ii_sup >= self.img_shape[0], jj_sup >= self.img_shape[1])
            ind_to_remove = np.arange(len(ii_sup))[np.reshape(ind_to_remove, (-1,))]
            ii_sup = np.reshape(np.delete(ii_sup, ind_to_remove), (-1, 1))
            jj_sup = np.reshape(np.delete(jj_sup, ind_to_remove), (-1, 1))
            ii_basis_centers = np.concatenate([ii_basis_centers, ii_sup])
            jj_basis_centers = np.concatenate([jj_basis_centers, jj_sup])

        box = multivariate_normal_pdf_2D(scale)
        ni_box, nj_box = box.shape
        ni_box = int((ni_box - 1) / 2)
        nj_box = int((nj_box - 1) / 2)
        ni, nj = self.img_shape

        # We compute the values of all pixels for each basis function
        for j in range(len(ii_basis_centers)):
            i_bf = int(ii_basis_centers[j])
            j_bf = int(jj_basis_centers[j])

            # If basis function belongs to the narrow band
            if narrow_band[i_bf, j_bf]:

                istart = max(0, i_bf - ni_box)
                iend = min(istart + (i_bf - istart) + ni_box + 1, ni)

                ibox_start = max(0, ni_box - i_bf)
                ibox_end = ibox_start + (iend - istart)

                jstart = max(0, j_bf - nj_box)
                jend = min(jstart + (j_bf - jstart) + nj_box + 1, nj)

                jbox_start = max(0, nj_box - j_bf)
                jbox_end = jbox_start + (jend - jstart)

                values = box[ibox_start:ibox_end, jbox_start:jbox_end]
                flatten_ind = flatten_indices[istart:iend, jstart:jend]

                non_zeros = np.where(values > 0)
                values = values[non_zeros]
                flatten_ind = flatten_ind[non_zeros]

                bf_values = scipy.sparse.coo_matrix((values, (flatten_ind, np.zeros(flatten_ind.shape))), shape=[self.N_tot, 1])

                basis_functions.append(bf_values)
                icoord.append(i_bf)
                jcoord.append(j_bf)

        basis_functions = scipy.sparse.hstack(basis_functions)

        return basis_functions, icoord, jcoord

    def createBasisFunctions_3D(self, narrow_band):

        flatten_indices = np.arange(self.N_tot)
        flatten_indices = np.reshape(flatten_indices, self.img_shape)

        basis_functions = []
        zcoord = []
        icoord = []
        jcoord = []

        scale = [self.config["spatial_prior_params"]["zscale"],
                 self.config["spatial_prior_params"]["iscale"],
                 self.config["spatial_prior_params"]["jscale"]]
        zstep, istep, jstep = self.config["spatial_prior_params"]["zstep"], \
                              self.config["spatial_prior_params"]["istep"], \
                              self.config["spatial_prior_params"]["jstep"]
        layout = self.config["spatial_prior_params"]["layout"]
        assert (layout == "square" or layout == "quinconce")

        zz_basis_centers = np.arange(self.img_shape[0])
        ii_basis_centers = np.arange(self.img_shape[1])
        jj_basis_centers = np.arange(self.img_shape[2])
        zz_basis_centers = zz_basis_centers % zstep
        zz_basis_centers = np.where(zz_basis_centers == 0)[0]
        ii_basis_centers = ii_basis_centers % istep
        ii_basis_centers = np.where(ii_basis_centers == 0)[0]
        jj_basis_centers = jj_basis_centers % jstep
        jj_basis_centers = np.where(jj_basis_centers == 0)[0]

        if layout == "quinconce":
            zz_basis_centers = zz_basis_centers[0::2]
            ii_basis_centers = ii_basis_centers[0::2]
            jj_basis_centers = jj_basis_centers[0::2]

        zz_basis_centers, ii_basis_centers, jj_basis_centers = np.meshgrid(zz_basis_centers, ii_basis_centers,
                                                                           jj_basis_centers, indexing="ij")
        zz_basis_centers, ii_basis_centers, jj_basis_centers = np.reshape(zz_basis_centers, (-1, 1)), np.reshape(
            ii_basis_centers, (-1, 1)), np.reshape(jj_basis_centers, (-1, 1))

        if layout == "quinconce":
            zz_sup = (zz_basis_centers + zstep).astype(np.int)
            ii_sup = (ii_basis_centers + istep).astype(np.int)
            jj_sup = (jj_basis_centers + jstep).astype(np.int)
            ind_to_remove = np.logical_or(ii_sup >= self.img_shape[1], jj_sup >= self.img_shape[2])
            ind_to_remove = np.logical_or(ind_to_remove, zz_sup >= self.img_shape[0])
            ind_to_remove = np.arange(len(ii_sup))[np.reshape(ind_to_remove, (-1,))]
            zz_sup = np.reshape(np.delete(zz_sup, ind_to_remove), (-1, 1))
            ii_sup = np.reshape(np.delete(ii_sup, ind_to_remove), (-1, 1))
            jj_sup = np.reshape(np.delete(jj_sup, ind_to_remove), (-1, 1))
            zz_basis_centers = np.concatenate([zz_basis_centers, zz_sup])
            ii_basis_centers = np.concatenate([ii_basis_centers, ii_sup])
            jj_basis_centers = np.concatenate([jj_basis_centers, jj_sup])

        box = multivariate_normal_pdf_3D(scale)
        nz_box, ni_box, nj_box = box.shape
        nz_box = int((nz_box - 1) / 2)
        ni_box = int((ni_box - 1) / 2)
        nj_box = int((nj_box - 1) / 2)
        nz, ni, nj = self.img_shape

        # We compute the values of all pixels for each basis function
        for j in range(len(ii_basis_centers)):
            z_bf = int(zz_basis_centers[j])
            i_bf = int(ii_basis_centers[j])
            j_bf = int(jj_basis_centers[j])

            # If basis function belongs to the narrow band
            if narrow_band[z_bf, i_bf, j_bf]:

                zstart = max(0, z_bf - nz_box)
                zend = min(zstart + (z_bf - zstart) + nz_box + 1, nz)

                zbox_start = max(0, nz_box - z_bf)
                zbox_end = zbox_start + (zend - zstart)

                istart = max(0, i_bf - ni_box)
                iend = min(istart + (i_bf - istart) + ni_box + 1, ni)

                ibox_start = max(0, ni_box - i_bf)
                ibox_end = ibox_start + (iend - istart)

                jstart = max(0, j_bf - nj_box)
                jend = min(jstart + (j_bf - jstart) + nj_box + 1, nj)

                jbox_start = max(0, nj_box - j_bf)
                jbox_end = jbox_start + (jend - jstart)

                flatten_ind = flatten_indices[zstart:zend, istart:iend, jstart:jend]
                values = box[zbox_start:zbox_end, ibox_start:ibox_end, jbox_start:jbox_end]

                non_zeros = np.where(values > 0)
                flatten_ind = flatten_ind[non_zeros]
                values = values[non_zeros]

                bf_values = scipy.sparse.coo_matrix((values, (flatten_ind, np.zeros(flatten_ind.shape))), shape=[self.N_tot, 1])

                basis_functions.append(bf_values)
                zcoord.append(z_bf)
                icoord.append(i_bf)
                jcoord.append(j_bf)

        basis_functions = scipy.sparse.hstack(basis_functions)

        return basis_functions, zcoord, icoord, jcoord

    def initialization(self):
        '''
        :return: alpha, mu_w
        '''

        alpha = torch.ones(1, dtype=float) * self.config["alpha0"]
        mu_w = torch.zeros(self.L, dtype=float) # L

        return alpha, mu_w

    def data_error(self, bf_mu_w, X):
        '''
        :param bf_mu_w: N
        :param X: N (appearance probability ratio)
        :return: data error, expit_bf_mu_w
        '''

        y = torch.sigmoid(bf_mu_w) # N

        # Handle probability zero cases
        if 1 in (y == 0) & (X == 1) or 1 in (y == 1) & (X == 0):
            # Error infinite when model give zero probability in contradiction to data
            e = torch.ones(1, dtype=float) * np.PINF

        else:
            e = - torch.sum(torch.log(X * y + (1 - X) * (1 - y)))

        return e, y

    def compute_beta(self, X, expit_bf_mu_w):
        '''
        :param X: N
        :param expit_bf_mu_w: N
        :return: beta
        '''

        z = (2 * X - 1) * expit_bf_mu_w + 1 - X
        a = 2 * (2 * X - 1) * expit_bf_mu_w * (1 - expit_bf_mu_w) ** 2
        b = (2 * X - 1) * expit_bf_mu_w * (1 - expit_bf_mu_w)
        c = ((2 * X - 1) * expit_bf_mu_w * (1 - expit_bf_mu_w)) ** 2
        beta = a / z - b / z - c / z ** 2

        return beta

    def posteriorMode(self, basis_functions_csr, basis_functions, mu_w, X, alpha):
        '''
        :param basis_functions_csr: N x L
        :param basis_functions: N x L
        :param mu_w: L
        :param X: N
        :param alpha: scalar
        '''

        cholesky = 1
        A = torch.eye(self.L, dtype=float) * alpha  # L x L

        step_min = 1 / (2 ** 8)  # Minimum fraction of the full Newton step considered

        # Get current model output and data error
        bf_mu_w = torch.sparse.mm(basis_functions, mu_w.view(-1, 1)).view((-1,)) # N
        dataError, expit_bf_mu_W = self.data_error(bf_mu_w, X)

        # Add the weight penalty
        regulariser = 1/2 * alpha * torch.sum(mu_w**2)
        newTotalError = dataError + regulariser

        for iteration in range(self.nr_n_iter):

            # Store the error at each iteration
            errorLog = newTotalError
            # Construct the gradient
            b = (2 * X - 1) * expit_bf_mu_W + 1 - X # N
            f_prime = (2 * X - 1) * expit_bf_mu_W * (1 - expit_bf_mu_W) * 1 / b # N

            f_prime_bf = torch.sparse.mm(basis_functions.transpose(1,0), f_prime.view(-1, 1)).view((-1,)) # L
            g = f_prime_bf - alpha * mu_w # L

            beta = - self.compute_beta(X, expit_bf_mu_W) # N

            # Compute the Hessian
            bf_bf = torch.zeros((self.L, self.L), dtype=float)
            tile = np.arange(0, self.N, 500000)
            for i in range(len(tile)):
                if i < len(tile) - 1:
                    bf = basis_functions_csr[tile[i]:tile[i + 1], :].toarray()  # N x L
                    b = beta[tile[i]:tile[i + 1]]
                else:
                    bf = basis_functions_csr[tile[i]:, :].toarray()
                    b = beta[tile[i]:]
                bf = torch.from_numpy(bf)
                bf_bf += torch.einsum('nl,n,no->lo', bf, b, bf)  # L x L
            H = bf_bf + A # L x L

            # Invert Hessian via Cholesky
            try:
                U = torch.cholesky(H, upper=True)
            except:
                cholesky = 0
                return 0, cholesky

            # Before processing, check for termination based on the gradient norm
            if torch.all(torch.abs(g) < self.nr_tol):
                break

            # If all OK, compute full Newton step H**(-1) * g
            Delta_mu_w = torch.einsum('lk,k->l', torch.cholesky_inverse(U, upper=True), g) # L

            step = 1

            while step > step_min:

                # Follow gradient to get new value of parameters
                mu_w_new = mu_w + step * Delta_mu_w
                bf_mu_w = torch.sparse.mm(basis_functions, mu_w_new.view(-1, 1)).view((-1,)) # N

                # Compute output and error at new point
                dataError, expit_bf_mu_W = self.data_error(bf_mu_w, X)
                regulariser = 1/2 * alpha * torch.sum(mu_w_new**2)
                newTotalError = dataError + regulariser

                # Test that we haven't made things worse
                if newTotalError >= errorLog:
                    # If so, back off!!
                    step = step / 2

                else:
                    mu_w = mu_w_new
                    step = 0  # This will force exit from the while loop

            # If we get here with non-zero step, it means that the smallest offset
            # from the current point along the "downhill" direction did not lead to
            # a decrease in error. In other words, we must be infinitesimally close
            # to a minimum (which is OK). So: break.

            if step != 0:
                break

        return (mu_w, U, beta, -dataError), cholesky

    def fullStatistics(self, basis_functions_csr, basis_functions, X, mu_w, alpha):
        '''
        :param basis_functions_csr: L x L
        :param basis_functions: N x L
        :param X: N
        :param mu_w: L
        :param alpha: scalar
        '''

        val, cholesky = self.posteriorMode(basis_functions_csr, basis_functions, mu_w, X, alpha)
        if cholesky == 0:
            return (0,), cholesky
        else:
            mu_w, U, beta, dataError = val

        sigma_w = torch.cholesky_inverse(U, upper=True)

        # Log marginal likelihood
        logdetHOver2 = torch.sum(torch.log(torch.diag(U)))
        bf_mu_w = torch.sparse.mm(basis_functions, mu_w.view(-1, 1)).view((-1,)) # N
        dataError, expit_bf_mu_W = self.data_error(bf_mu_w, X)
        logML = - dataError - 1 / 2 * torch.sum((mu_w** 2) * alpha) + self.L/2 * torch.log(alpha) - logdetHOver2

        return (mu_w, sigma_w, logML), cholesky

    def alpha_update(self, mu_w, sigma_w, alpha):
        '''
        :param mu_w: L
        :param sigma_w: L x L
        :param alpha: scalar
        :return: alpha
        '''

        alpha = (self.L - alpha * torch.trace(sigma_w)) / torch.sum(mu_w**2)

        return alpha

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

        # Create the basis functions
        if self.dim == 2:
            basis_functions, icoord, jcoord = self.createBasisFunctions_2D(narrow_band)
        else:
            basis_functions, zcoord, icoord, jcoord = self.createBasisFunctions_3D(narrow_band)
        self.L = basis_functions.shape[1]  # Number of basis functions

        basis_functions_csr = basis_functions.tocsr()
        basis_functions_csr = basis_functions_csr[np.reshape(narrow_band, (-1,)), :]  # N x L

        basis_functions = basis_functions_csr.tocoo()
        index = np.concatenate([np.reshape(basis_functions.row, (1, -1)),
                                np.reshape(basis_functions.col, (1, -1))], axis=0).astype(int)
        basis_functions = torch.sparse.FloatTensor(torch.from_numpy(index), torch.from_numpy(basis_functions.data),
                                                   torch.Size([self.N, self.L]))

        # Initialization
        appearance_ratio = torch.from_numpy(appearance_ratio)
        alpha, mu_w = self.initialization()

        log_ML_list = []
        i = 0
        converged = 0

        # Do a first iteration
        val, cholesky = self.fullStatistics(basis_functions_csr, basis_functions, appearance_ratio, mu_w, alpha)
        if cholesky:
            (mu_w, sigma_w, logML) = val
            log_ML_list.append(logML)
        else:
            raise Exception("Cholesky failed")

        # Save values in case cholesky fails later
        alpha_copy = torch.empty_like(alpha, dtype=float)
        alpha_copy.data = alpha.clone()
        sigma_w_copy = torch.empty_like(sigma_w, dtype=float)
        sigma_w_copy.data = sigma_w.clone()
        mu_w_copy = torch.empty_like(mu_w, dtype=float)
        mu_w_copy.data = mu_w.clone()

        while i < self.n_iter:

            alpha_old = torch.empty_like(alpha, dtype=float)
            alpha_old.data = alpha.clone()
            alpha = self.alpha_update(mu_w, sigma_w, alpha)

            if i > 1:
                diff = alpha_old - alpha
                if torch.abs(diff / alpha_old) < self.tol:
                    i = self.n_iter
                    converged = 1

            val, cholesky = self.fullStatistics(basis_functions_csr, basis_functions, appearance_ratio, mu_w, alpha)
            if cholesky:
                (mu_w, sigma_w, logML) = val
                log_ML_list.append(logML)

                # Save values in case cholesky fails later
                alpha_copy = torch.empty_like(alpha, dtype=float)
                alpha_copy.data = alpha.clone()
                sigma_w_copy = torch.empty_like(sigma_w, dtype=float)
                sigma_w_copy.data = sigma_w.clone()
                mu_w_copy = torch.empty_like(mu_w, dtype=float)
                mu_w_copy.data = mu_w.clone()

            else:
                mu_w = mu_w_copy
                sigma_w = sigma_w_copy
                alpha = alpha_copy
                i = self.n_iter

            i += 1

        # Check convergence
        if not converged and self.n_iter > 0:
            self.converged = False
        else:
            self.converged = True

        self.log_ML_list = [float(i.cpu()) for i in log_ML_list]
        self.mu_w = mu_w.cpu().numpy()
        self.sigma_w = sigma_w.cpu().numpy()
        self.alpha = alpha.cpu().numpy()
        bf_mu_w = torch.sparse.mm(basis_functions, mu_w.view(-1, 1)).view((-1,)) # N
        self.bf_mu_w = bf_mu_w.cpu().numpy()  # N
        self.basis_functions = basis_functions_csr

        # Compute posterior
        expit_bf_mu_w = torch.sigmoid(bf_mu_w)
        a = appearance_ratio * expit_bf_mu_w + (1 - appearance_ratio) * (1 - expit_bf_mu_w)
        b = appearance_ratio * expit_bf_mu_w  # P(I|Z)*P(Z|W)
        rn = b / a # N
        rn = torch.cat([rn.view((-1,1)), (1 -rn).view((-1,1))], dim=1)
        self.rn = rn.cpu().numpy()

        self.fitted = True

        return self


if __name__ =="__main__":

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from sklearn.mixture import BayesianGaussianMixture
    from scipy.special import expit

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
    spatial_prior_params = {
        "iscale": 40,
        "jscale": 40,
        "istep": 20,
        "jstep": 20,
        "layout": "square",
    }
    config = {
        "dim": 2,
        "tol": 1e-3,
        "eps": 1e-8,
        "n_iter": 100,
        "spatial_prior_params": spatial_prior_params,
        "alpha0": 1e-3,
        "nr_tol": 1e-5,
        "nr_n_iter": 25,
    }
    model = GLSP_regularization(config)

    # Appearance model
    appearance_model = BayesianGaussianMixture(n_components=2)
    appearance_model.fit(np.reshape(img, (-1,1)))
    X = appearance_model.predict_proba(np.reshape(img, (-1,1)))[:,0]

    # Inputs
    narrow_band = np.ones(img.shape, dtype=np.bool)
    X = np.reshape(X, (-1,))

    model.fit(X, narrow_band)

    # Plot the log marginal likelihood
    plt.figure(figsize=[3,3])
    plt.plot(model.log_ML_list)
    plt.xlabel("Iterations")
    plt.ylabel("log ML")
    plt.show()

    # Plot the result
    rn = model.rn[:,0]
    bf_mu_w = expit(model.bf_mu_w)
    plt.figure(figsize=[3,3])
    plt.imshow(img, cmap="gray")
    handles= []
    plt.contour(np.reshape(rn, img.shape), levels=[0.5], colors='red', linewidths=2)
    handles.append(Line2D([], [], color='red', linestyle="-", label='P(I|Z)', linewidth=2))
    plt.contour(np.reshape(bf_mu_w, img.shape), levels=[0.5], colors='green', linewidths=2)
    handles.append(Line2D([], [], color='green', linestyle="-", label='P(Z|W)', linewidth=2))
    plt.contour(np.reshape(X, img.shape), levels=[0.5], colors='blue', linewidths=2)
    handles.append(Line2D([], [], color='blue', linestyle="-", label='P(Z|I,W)', linewidth=2))
    plt.legend(handles=handles)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
