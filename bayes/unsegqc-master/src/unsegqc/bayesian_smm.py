"""t-Student Mixture Models Module (smm).

- This module allows you to model data by a mixture of t-Student distributions, estimating the parameters with
variational inference. It is an implementation of the paper:
                    Archambeau C, Verleysen M. Robust Bayesian clustering.
                    Neural Netw. 2007;20(1):129-138. doi:10.1016/j.neunet.2006.06.009

- This module has reused code and comments from sklearn.mixture.gmm.  
"""

import numpy as np
import sklearn
import sklearn.cluster
import sklearn.utils
import scipy.linalg
import scipy.special
import scipy.optimize
import warnings


class dofMaximizationError(ValueError):
    def __init__(self, message):
        super(dofMaximizationError, self).__init__(message)


class SMM(sklearn.base.BaseEstimator):
    """t-Student Mixture Model SMM class.

    Representation of a t-Student mixture model probability 
    distribution. This class allows for easy evaluation of, sampling
    from, and maximum-likelihood estimation of the parameters of an 
    SMM distribution.

    Initializes parameters such that every mixture component has 
    zero mean and identity covariance.

    Parameters
    ----------
    n_components : int, optional.
                   Number of mixture components. 
                   Defaults to 1.

    random_state: RandomState or an int seed.
                  A random number generator instance. 
                  None by default.

    tol : float, optional.
          Convergence threshold. EM iterations will stop when 
          average gain in log-likelihood is below this threshold.  
          Defaults to 1e-6.

    min_covar : float, optional.
                Floor on the diagonal of the covariance matrix to 
                prevent overfitting. 
                Defaults to 1e-6.

    n_iter : int, optional.
             Number of EM iterations to perform. 
             Defaults to 1000.

    n_init : int, optional.
             Number of initializations to perform. The best result 
  			  is kept.
             Defaults to 1.

    Attributes
    ----------
    weights_ : array, shape (`n_components`,).
               This attribute stores the mixing weights for each 
               mixture component.

    means_ : array_like, shape (`n_components`, `n_features`).
             Mean parameters for each mixture component.

    covars_ : array_like, shape (`n_components`, `n_features`, `n_features`)
              Covariance parameters for each mixture component.

    converged_ : bool.
                 True when convergence was reached in fit(), False 
                 otherwise.
    """

    def __init__(self, n_components=1, random_state=None, tol=1e-6, min_covar=1e-6,
                 n_iter=1000, n_init=1):

        # Store the parameters as class attributes
        self.n_components = n_components
        self.random_state = random_state
        self.tol = tol
        self.min_covar = min_covar
        self.n_iter = n_iter
        self.n_init = n_init

    def _expectation_step(self, X):
        """Performs the expectation step of the VEM algorithm.

        This method uses the means, class-related weights, 
        covariances and degrees of freedom stored in the attributes 
        of this class: 
        self.means_, self.weights_, self.covars_, and self.degrees_.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features). 
            Matrix with all the data points, each row represents a 
            new data point and the columns represent each one of the
            features.
        """

        # Sanity checks:
        #    - Check that the fit() method has been called before this 
        #      one.
        #    - Convert input to 2d array, raise error on sparse 
        #      matrices. Calls assert_all_finite by default.
        #    - Check that the the X array is not empty of samples.
        #    - Check that the no. of features is equivalent to the no. 
        #      of means that we have in self.
        sklearn.utils.validation.check_is_fitted(self, 'means_')
        X = sklearn.utils.validation.check_array(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[:, np.newaxis]
        if X.size == 0:
            return np.array([]), np.empty((0, self.n_components))
        if X.shape[1] != self.means_.shape[1]:
            raise ValueError(
                '[SMM._expectation_step] Error, the ' \
                + 'shape of X is not compatible with self.'
            )

        # Initialisation of reponsibilities and weight of each point for
        # the Gamma distribution
        n_samples, n_dim = X.shape
        responsibilities = np.ndarray(shape=(X.shape[0], self.n_components),
                                      dtype=np.float64
                                      )
        gammaweights_ = np.ndarray(shape=(X.shape[0], self.n_components),
                                   dtype=np.float64
                                   )

        # Weights
        kappa_sum = np.sum(self.kappa_)
        weights = np.exp(scipy.special.digamma(self.kappa_) - scipy.special.digamma(kappa_sum))

        # Calculate the probability of each point belonging to each 
        # t-Student distribution of the mixture
        pr_before_weighting = self._multivariate_t_student_density_full(
            X, self.means_, self.covars_, self.degrees_, self.gamma_, self.neta_,
            self.min_covar
        )
        pr = pr_before_weighting * weights

        # Calculate the likelihood of each point
        likelihoods = pr.sum(axis=1)

        # Update responsibilities
        responsibilities = \
            pr / (likelihoods.reshape(likelihoods.shape[0], 1)
                  + 10 * SMM._EPS
                  )

        # Update the Gamma weight for each observation
        vp = self.degrees_ + n_dim
        maha_dist = SMM._mahalanobis_distance_mix_full(X, self.means_, self.covars_, self.min_covar)
        gammaweights_ = vp / (self.degrees_ + self.gamma_ * maha_dist + n_dim / self.neta_)

        return likelihoods, responsibilities, gammaweights_, pr

    def _maximisation_step(self, X, responsibilities, gammaweights_):
        """Perform the maximisation step of the EM algorithm.
        
        Parameters
        ----------
        X : array_like, shape (n_samples, n_features).              
            Each row corresponds to a single data point.

        responsibilities : array_like, shape (n_samples, n_components). 

        gammaweights_ : array_like, shape (n_samples, n_components).
        """

        n_samples, n_dim = X.shape
        z_sum = responsibilities.sum(axis=0)
        zu = responsibilities * gammaweights_
        zu_sum = zu.sum(axis=0)

        # Update neta
        self.neta_ = np.reshape(zu_sum, (self.n_components,)) + self.neta_0

        # Update gamma
        self.gamma_ = np.reshape(z_sum, (self.n_components,)) + self.gamma_0

        # Update kappa
        self.kappa_ = np.reshape(z_sum, (self.n_components,)) + self.kappa_0

        # Update means
        dot_zu_x = np.dot(zu.T, X)
        omega_bar = np.reshape(1 / n_samples * zu_sum, (self.n_components, 1))
        mu_bar = dot_zu_x * 1 / (n_samples * omega_bar)
        neta0 = np.reshape(self.neta_0, (self.n_components, 1))
        neta = np.reshape(self.neta_, (self.n_components, 1))
        self.means_ = (mu_bar * n_samples * omega_bar + neta0 * self.mean_0) / (neta + 10 * SMM._EPS)

        # Update covariances
        covars = SMM._covar_mstep_full(X, zu, self.means_, self.min_covar)

        tmp = np.empty(covars.shape)
        for i in range(self.n_components):
            mu_bar_i = np.reshape(mu_bar[i], (-1, 1))
            diff = mu_bar_i - self.mean_0

            tmp[i] = n_samples * omega_bar[i] * covars[i] + \
                     zu_sum[i] * self.neta_0[i] / self.neta_[i] \
                     * np.dot(diff, diff.T) + self.covar_0

        self.covars_ = tmp

        # Update degrees of freedom
        try:
            self.degrees_ = SMM._solve_dof_equation(
                self.degrees_, responsibilities, z_sum,
                gammaweights_, n_dim, self.tol, self.n_iter
            )
        except FloatingPointError as fpe:
            message = str(fpe)
            raise dofMaximizationError(message)
        except RuntimeError as re:
            message = str(re)
            if message.startswith('Failed to converge after'):
                warnings.warn(message, RuntimeWarning)
                pass

    def fit(self, X, y=None):
        """Estimate model parameters with the EM algorithm.

        A initialization step is performed before entering the em
        algorithm. If you want to avoid this step, set the keyword
        argument init_params to the empty string '' when creating 
        the SMM object. Likewise, if you would like just to do an
        initialization, set n_iter=0.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features).

        y : not used, just for compatibility with sklearn API.
        """

        # Sanity check: assert that the input matrix is not 1D
        if (len(X.shape) == 1):
            raise ValueError(
                '[SMM.fit] Error, the input matrix must have a ' \
                + 'shape of (n_samples, n_features).'
            )

        # Sanity checks:
        #    - Convert input to 2d array, raise error on sparse 
        #      matrices. Calls assert_all_finite by default.
        #    - No. of samples is higher or equal to the no. of 
        #      components in the mixture.
        X = sklearn.utils.validation.check_array(X, dtype=np.float64)
        if X.shape[0] < self.n_components:
            raise ValueError(
                '[SMM.fit] Error, SMM estimation with ' \
                + '%s components, but got only %s samples' % (
                    self.n_components, X.shape[0]
                )
            )

        # For all the initialisations we get the one with the best 
        # parameters
        n_samples, n_dim = X.shape
        max_prob = -np.infty
        for _ in range(self.n_init):

            # Initialization of prior parameters
            self.kappa_0 = None
            self.neta_0 = None
            self.mean_0 = None
            self.covar_0 = None
            self.gamma_0 = None
            self._check_parameters(X)
            self.degrees_ = np.tile(1.0, self.n_components)

            # First M-step with kmean initialization
            kmeans = sklearn.cluster.KMeans(
                n_clusters=self.n_components,
                init='k-means++',
                random_state=self.random_state
            )
            resp = np.zeros((n_samples, self.n_components))
            label = kmeans.fit(X).labels_
            resp[np.arange(n_samples), label] = 1

            gammaweights_ = np.ones(resp.shape)

            self._maximisation_step(X, resp,
                                    gammaweights_
                                    )

            best_params = {
                'kappa': self.kappa_,
                'neta': self.neta_,
                'gamma': self.gamma_,
                'means': self.means_,
                'covars': self.covars_,
                'degrees': self.degrees_
            }

            self.converged_ = False
            current_log_likelihood = None

            # VEM algorithm
            for i in range(self.n_iter):
                prev_log_likelihood = current_log_likelihood

                # Expectation step
                likelihoods, responsibilities, gammaweights_, _ = \
                    self._expectation_step(X)

                # Sanity check: assert that the likelihoods, 
                # responsibilities and gammaweights have the correct
                # dimensions
                assert (len(likelihoods.shape) == 1)
                assert (likelihoods.shape[0] == n_samples)
                assert (len(responsibilities.shape) == 2)
                assert (responsibilities.shape[0] == n_samples)
                assert (responsibilities.shape[1] == self.n_components)
                assert (len(gammaweights_.shape) == 2)
                assert (gammaweights_.shape[0] == n_samples)
                assert (gammaweights_.shape[1] == self.n_components)

                # Calculate loss function
                current_log_likelihood = np.log(likelihoods).mean()

                # Check for convergence
                if prev_log_likelihood is not None:
                    change = np.abs(current_log_likelihood -
                                    prev_log_likelihood
                                    )
                    if change < self.tol:
                        self.converged_ = True
                        break

                # Maximisation step
                try:
                    self._maximisation_step(X, responsibilities,
                                            gammaweights_
                                            )
                except dofMaximizationError as e:
                    print(
                        '[self._maximisation_step] Error in the ' \
                        + 'maximization step of the degrees of ' \
                        + 'freedom: ' + e.message
                    )
                    break

            # If the results are better, keep it
            if self.n_iter and self.converged_:
                if current_log_likelihood > max_prob:
                    max_prob = current_log_likelihood
                    best_params = {
                        'kappa': self.kappa_,
                        'neta': self.neta_,
                        'gamma': self.gamma_,
                        'means': self.means_,
                        'covars': self.covars_,
                        'degrees': self.degrees_
                    }

        # Check the existence of an init param that was not subject to
        # likelihood computation issue
        self.fitted = True
        if np.isneginf(max_prob) and self.n_iter:
            msg = 'EM algorithm was never able to compute a valid ' \
                  + 'likelihood given initial parameters. Try ' \
                  + 'different init parameters (or increasing ' \
                  + 'n_init) or check for degenerate data.'
            warnings.warn(msg, RuntimeWarning)
            self.fitted = False

        # Choosing the best result of all the iterations as the actual 
        # result
        if self.n_iter:

            self.kappa_ = best_params['kappa']
            self.neta_ = best_params['neta']
            self.gamma_ = best_params['gamma']
            self.means_ = best_params['means']
            self.covars_ = best_params['covars']
            self.degrees_ = best_params['degrees']

            # Compute mean, weight 
            self.mu_ = self.means_
            self.sigma_ = np.empty(self.covars_.shape)
            for i in range(self.n_components):
                self.sigma_[i] = self.covars_[i] / self.gamma_[i]
            self.weights_ = self.kappa_ / np.sum(self.kappa_)

        return self

    def _check_parameters(self, X):
        """Check that the parameters are well defined.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        """

        self._check_weights_parameters()
        self._check_means_parameters(X)
        self._checkcovariance_prior_parameter(X)
        self._check_gamma_parameters(X)

    def _check_weights_parameters(self):
        """Check the parameter of the Dirichlet distribution."""

        if self.kappa_0 is None:
            self.kappa_0 = 1. / self.n_components
        elif self.kappa_0 > 0.:
            self.kappa_0 = (
                self.kappa_0)
        else:
            raise ValueError("The parameter 'weight_concentration_prior' "
                             "should be greater than 0., but got %.3f."
                             % self.kappa_0)

    def _check_means_parameters(self, X):
        """Check the parameters of the Gaussian distribution.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        """
        _, n_features = X.shape

        if self.neta_0 is None:
            self.neta_0 = np.ones(self.n_components)
        elif self.neta_0 > 0.:
            self.neta_0 = self.neta_0
        else:
            raise ValueError("The parameter 'mean_precision_prior' should be "
                             "greater than 0., but got %.3f."
                             % self.neta_0)

        if self.mean_0 is None:
            self.mean_0 = X.mean(axis=0)

    def _check_gamma_parameters(self, X):
        """Check the prior parameters of the gamma distribution.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        """
        _, n_features = X.shape

        if self.gamma_0 is None:
            self.gamma_0 = n_features
        elif self.gamma_0 > n_features - 1.:
            self.gamma_0 = self.gamma_0
        else:
            raise ValueError("The parameter 'degrees_of_freedom_prior' "
                             "should be greater than %d, but got %.3f."
                             % (n_features - 1, self.gamma_0))

    def _checkcovariance_prior_parameter(self, X):
        """Check the `covariance_prior_`.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        """
        _, n_features = X.shape

        if self.covar_0 is None:
            self.covar_0 = np.atleast_2d(np.cov(X.T))

    def predict(self, X):
        """Predict label for data.

        This function will tell you which component of the mixture
        most likely generated the sample.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features).

        Returns
        -------
        r_argmax : array_like, shape (n_samples,). 
        """

        _, responsibilities, _, _ = self._expectation_step(X)
        r_argmax = responsibilities.argmax(axis=1)

        return r_argmax

    def predict_proba(self, X):
        """Predict label for data.

        This function will tell the probability of each component
        generating each sample.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features).

        Returns
        -------
        responsibilities : array_like, shape (n_samples, n_components).
        """

        _, responsibilities, _, _ = self._expectation_step(X)

        return responsibilities

    def predict_mixture_density(self, X):

        _, _, _, pr = self._expectation_step(X)

        return pr

    def score(self, X, y=None):
        """Compute the log probability under the model.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features). 

        Returns
        -------
        prob : array_like, shape (n_samples,). 
               Probabilities of each data point in X.
        """

        prob, _ = self.score_samples(X)

        return prob

    @staticmethod
    def _solve_dof_equation(v_vector, z, z_sum, u, n_dim, tol, n_iter):
        """Solves the equation to calculate the next value of v 
        (degrees of freedom).

        This method is part of the maximisation step. It is used to 
        calculate the next value for the degrees of freedom of each 
        t-Student component. This method uses the information from 
        the E-step as well as the number of dimensions (features) of
        a point.

        Parameters
        ----------
        v_vector : array_like, shape (n_components,).
                   Degrees of freedoom of ALL the components of the 
                   mixture.

        z : array_like, shape (n_samples, n_components).
            Matrix of responsibilities, each row represents a point 
            and each column represents a component of the mixture.

        z_sum : array_like, shape (n_samples,).  
                Sum of all the rows of the matrix of 
                responsibilities.

        u : array_like, shape (n_samples, n_components). 
            Matrix of gamma weights, each row represents a point and
            each column represents a component of the mixture.
    
        n_dim : integer. 
                Number of features of each data point.
        
        Returns
        -------
        new_v_vector : array_like (n_components,).
                       Vector with the updated degrees of freedom for 
                       each component in the mixture.
        """

        # Sanity check: check that the dimensionality of the vector of 
        # degrees of freedom is correct
        assert (len(v_vector.shape) == 1)
        n_components = v_vector.shape[0]

        # Sanity check: the matrix of responsibilities should be 
        # (n_samples x n_components)
        assert (len(z.shape) == 2)
        assert (z.shape[1] == n_components)

        # Sanity check: the top-to-bottom sum of the responsibilities 
        # must have a shape (n_components, )
        assert (len(z_sum.shape) == 1)
        assert (z_sum.shape[0] == n_components)

        # Sanity check: gamma weights matrix must have the same 
        # dimensionality as the responsibilities
        assert (u.shape == z.shape)

        # Initialisation
        new_v_vector = np.empty_like(v_vector)

        # Calculate the constant part of the equation to calculate the 
        # new degrees of freedom
        vdim = (v_vector + n_dim) / 2.0
        zlogu_sum = np.sum(z * (np.log(u) - u), axis=0)
        constant_part = 1.0 \
                        + zlogu_sum / z_sum \
                        + scipy.special.digamma(vdim) \
                        - np.log(vdim)

        # Solve the equation numerically using Newton-Raphson for each 
        # component of the mixture
        for c in range(n_components):
            def func(x):
                return np.log(x / 2.0) \
                       - scipy.special.digamma(x / 2.0) \
                       + constant_part[c]

            def fprime(x):
                return 1.0 / x \
                       - scipy.special.polygamma(1, x / 2.0) / 2.0

            def fprime2(x):
                return - 1.0 / (x * x) \
                       - scipy.special.polygamma(2, x / 2.0) / 4.0

            new_v_vector[c] = scipy.optimize.newton(
                func, v_vector[c], fprime, args=(), tol=tol,
                maxiter=n_iter, fprime2=fprime2
            )
            if new_v_vector[c] < 0.0:
                raise ValueError('[_solve_dof_equation] Error, ' \
                                 + 'degree of freedom smaller than one. \n' \
                                 + 'n_components[c] = ' \
                                 + str(n_components) \
                                 + '. \n' + 'v_vector[c] = ' \
                                 + str(v_vector[c]) \
                                 + '. \n' \
                                 + 'new_v_vector[c] = ' \
                                 + str(new_v_vector[c]) \
                                 + '. \n' \
                                 + 'constant_part[c] = ' \
                                 + str(constant_part[c]) \
                                 + '. \n' \
                                 + 'zlogu_sum[c] = ' \
                                 + str(zlogu_sum[c]) \
                                 + '. \n' \
                                 + 'z_sum[c] = ' \
                                 + str(z_sum[c]) \
                                 + '. \n' \
                                 + 'z = ' + str(z) + '. \n'
                                 )

        return new_v_vector

    @staticmethod
    def _covar_mstep_full(X, zu, means, min_covar):
        """Performing the covariance m-step for full covariances.
    
        Parameters
        ----------
        X : array_like, shape (n_samples, n_features).
            List of k_features-dimensional data points. Each row 
            corresponds to a single 
            data point.

        zu : array_like, shape (n_samples, n_components).
             Contains the element-wise multiplication of the 
             responsibilities by the gamma weights.

        z_sum : array_like, shape (n_components,)
                Sum of all the responsibilities for each mixture.

        means : array_like, shape (n_components, n_features).
                List of n_features-dimensional mean vectors for 
                n_components t-Students.
                Each row corresponds to a single mean vector.

        min_covar : float value.
                    Minimum amount that will be added to the 
                    covariance matrix in case of trouble, usually 1.e-7.

        Returns
        -------
        cv : array_like, shape (n_components, n_features, 
             n_features).
             New array of updated covariance matrices.
        """

        # Sanity checks for dimensionality
        n_samples, n_features = X.shape
        n_components = means.shape[0]
        assert (zu.shape[0] == n_samples)
        assert (zu.shape[1] == n_components)

        # Eq. 31 from D. Peel and G. J. McLachlan, "Robust mixture 
        # modelling using the t distribution"
        cv = np.empty((n_components, n_features, n_features))
        zu_sum = zu.sum(axis=0)
        for c in range(n_components):
            post = zu[:, c]
            mu = means[c]
            diff = X - mu
            with np.errstate(under='ignore'):
                # Underflow Errors in doing post * X.T are not important
                avg_cv = np.dot(post * diff.T, diff) \
                         / (zu_sum[c] + 10 * SMM._EPS)

            cv[c] = avg_cv + min_covar * np.eye(n_features)

        return cv

    @staticmethod
    def _multivariate_t_student_density_full(X, means, covars, dfs, gamma, neta, min_covar):
        """Multivariate t-Student PDF for a matrix of data points.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features). 
            Each row corresponds to a single data point.

        means : array_like, shape (n_components, n_features).
                Mean vectors for n_components t-Students.
                Each row corresponds to a single mean vector.

        covars : array_like, shape (n_components, n_features, 
                 n_features). 
                 Covariance parameters for each t-Student. 

        dfs : array_like, shape (n_components,).
              Degrees of freedom.

        min_covar : float value.
                    Minimum amount that will be added to the covariance 
                    matrix in case of trouble, usually 1.e-6.

        Returns
        -------
        prob : array_like, shape (n_samples, n_components).
               Evaluation of the multivariate probability density 
               function for a t-Student distribution.
        """

        n_samples, n_dim = X.shape
        n_components = len(means)
        prob = np.empty((n_samples, n_components))

        # Sanity check: assert that the received means and covars have 
        # the right shape
        assert (means.shape[0] == n_components)
        assert (covars.shape[0] == n_components)
        assert (dfs.shape[0] == n_components)
        assert (gamma.shape[0] == n_components)
        assert (neta.shape[0] == n_components)

        # We evaluate all the saples for each component 'c' in the 
        # mixture
        for c, (mu, cv, df, gamma_, neta_) in enumerate(zip(means, covars, dfs, gamma, neta)):

            # Calculate the Cholesky decomposition of the covariance 
            # matrix
            cov_chol = SMM._cholesky(cv, min_covar)

            # Calculate the determinant of the covariance matrix
            cov_det = np.power(np.prod(np.diagonal(cov_chol)), 2)
            sum_ = 0
            for i in range(1, n_dim + 1):
                sum_ += scipy.special.digamma((gamma_ + 1 - i) / 2)
            cov_det = np.exp(sum_) * 2 ** n_dim * cov_det ** (-1)

            # Calculate the Mahalanobis distance between each vector and
            # the mean
            maha = SMM._mahalanobis_distance_chol(X, mu, cov_chol)

            # Calculate the coefficient of the gamma functions
            r = np.asarray(df, dtype=np.float64)
            gamma_coef = np.exp(
                scipy.special.gammaln((r + n_dim) / 2.0) \
                - scipy.special.gammaln(r / 2.0)
            ) * np.sqrt(cov_det)  # The covariance matrix is the PRECISION matrix

            # Calculate the denominator of the multivariate t-Student
            # We use the log to avoid overflow problems
            # denom = np.power(np.pi * df, n_dim / 2.0) \
            #         * np.power((1 + maha*gamma_ / df + n_dim/(df*neta_)), (df + n_dim) / 2)
            denom = n_dim / 2 * np.log(np.pi * df) + (df + n_dim) / 2 * np.log(
                1 + maha * gamma_ / df + n_dim / (df * neta_))

            # Finally calculate the PDF of the class 'c' for all the X samples
            # prob[:, c] = gamma_coef / denom
            prob[:, c] = np.exp(np.log(gamma_coef) - denom)

        return prob

    @staticmethod
    def _cholesky(cv, min_covar):
        """Calculates the lower triangular Cholesky decomposition of a 
        covariance matrix.
        
        Parameters
        ----------
        covar : array_like, shape (n_features, n_features).
                Covariance matrix whose Cholesky decomposition wants to 
                be calculated.

        min_covar : float value.
                    Minimum amount that will be added to the covariance 
                    matrix in case of trouble, usually 1.e-6.

        Returns
        -------
        cov_chol : array_like, shape (n_features, n_features).
                   Lower Cholesky decomposition of a covariance matrix.
        """

        # Sanity check: assert that the covariance matrix is squared
        assert (cv.shape[0] == cv.shape[1])

        # Sanity check: assert that the covariance matrix is symmetric
        if (cv.transpose() - cv).sum() > min_covar:
            print('[SMM._cholesky] Error, covariance matrix not ' \
                  + 'symmetric: '
                  + str(cv)
                  )

        n_dim = cv.shape[0]
        try:
            cov_chol = scipy.linalg.cholesky(cv, lower=True)
        except scipy.linalg.LinAlgError:
            # The model is most probably stuck in a component with too
            # few observations, we need to reinitialize this components
            cov_chol = scipy.linalg.cholesky(
                cv + min_covar * np.eye(n_dim), lower=True
            )

        return cov_chol

    @staticmethod
    def _mahalanobis_distance_chol(X, mu, cov_chol):
        """Calculates the Mahalanobis distance between a matrix (set) of
        vectors (X) and another vector (mu).

        The vectors must be organised by row in X, that is, the features
        are the columns.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features).
            Sample in each row.

        mu : array_like (n_features).
             Mean vector of a single distribution (no mixture).

        cov_chol : array_like, shape (n_features, n_features).
                   Cholesky decomposition (L, i.e. lower triangular) of 
                   the covariance (normalising) matrix in case that is 
                   a full matrix. 
        
        Returns
        -------
        retval : array_like, shape (n_samples,).
                 Array of distances, each row represents the distance
                 from the vector in the same row of X and mu. 
        """

        z = scipy.linalg.solve_triangular(
            cov_chol, (X - mu).T, lower=True
        )
        retval = np.einsum('ij, ij->j', z, z)

        return retval

    @staticmethod
    def _mahalanobis_distance_mix_full(X, means, covars, min_covar):
        """Calculates the mahalanobis distance between a matrix of 
        points and a mixture of distributions. 
        
	Parameters
        ----------
        X : array_like, shape (n_samples, n_features).    
            Matrix with a sample vector in each row.

        means : array_like, shape (n_components, n_features).
                Each row corresponds to a single mean vector.

        covars : array_like, shape (n_components, n_features, 
                 n_features).
                 Covariance parameters for each t-Student. 

        Returns
        -------
        result : array_like, shape (n_samples, n_components).
                 Mahalanobis distance from all the samples to all the 
                 component means.
        """

        n_samples, n_dim = X.shape
        n_components = len(means)
        result = np.empty((n_samples, n_components))
        for c, (mu, cv) in enumerate(zip(means, covars)):
            cov_chol = SMM._cholesky(cv, min_covar)
            result[:, c] = SMM._mahalanobis_distance_chol(
                X, mu, cov_chol
            )

        return result

    @staticmethod
    def multivariate_t_rvs(m, S, df=np.inf, n=1):
        """Generate multivariate random variable sample from a t-Student
        distribution.
        
        Author
        ------
        Original code by Enzo Michelangeli.
        Modified by Luis C. Garcia-Peraza Herrera.
        This static method is exclusively used by 'tests/smm_test.py'.

        Parameters
        ----------
        m : array_like, shape (n_features,).
            Mean vector, its length determines the dimension of the 
            random variable.

        S : array_like, shape (n_features, n_features).
            Covariance matrix.

        df : int or float.
             Degrees of freedom.

        n : int. 
            Number of observations.

        Returns
        -------
        rvs : array_like, shape (n, len(m)). 
              Each row is an independent draw of a multivariate t 
              distributed random variable.
        """

        # Sanity check: dimension of mean and covariance compatible
        assert (len(m.shape) == 1)
        if (m.shape[0] != 1):
            assert (m.shape[0] == S.shape[1])
            assert (len(S.shape) == 2)
        assert (m.shape[0] == S.shape[0])

        # m = np.asarray(m)
        d = m.shape[0]
        # d = len(m)
        if df == np.inf:
            x = 1.0
        else:
            x = np.random.chisquare(df, n) / df

        z = np.random.multivariate_normal(np.zeros(d), S, (n,))
        retval = m + z / np.sqrt(x)[:, None]

        return retval

    @property
    def weights(self):
        """Returns the weights of each component in the mixture."""
        return self.weights_

    @property
    def means(self):
        """Returns the means of each component in the mixture."""
        return self.mu_

    @property
    def degrees(self):
        """Returns the degrees of freedom of each component in the 
        mixture."""
        return self.degrees_

    @property
    def covariances(self):
        """Covariance parameters for each mixture component.

        Returns
        -------
        The covariance matrices for all the classes. 
        The shape depends on the type of covariance matrix:

            (n_classes,  n_features)               if 'diag',
            (n_classes,  n_features, n_features)   if 'full'
            (n_classes,  n_features)               if 'spherical',
            (n_features, n_features)               if 'tied',
        """

        return self.sigma_

    _EPS = np.finfo(np.float64).eps

