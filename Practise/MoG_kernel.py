import GPy
from GPy.kern import Kern


# JC: This is my own MoG kernel EDIT THIS ONE


# -*- coding: utf-8 -*-
# Copyright (c) 2015, Alex Grigorevskiy
# Licensed under the BSD 3-clause license (see LICENSE.txt)
"""
The MoG kernel.
"""


from GPy.core.parameterization import Param
from paramz.transformations import Logexp

import numpy as np


class CustomMogKernel(Kern):
    """
    Custom MoG kernel

    .. math::

       k(x, y) = (1 / pi) * exp(-\theta / 2 * ||x - y||^2) * Σ_{f ∈ frequencies} Σ_{m=1}^{M} cos(m * 2πf * ||x - y||)

    :param input_dim: the number of input dimensions
    :type input_dim: int
    :param variance: the variance :math:`\theta_1` in the formula above
    :type variance: float
    :param frequency: the vector of frequencys :math:`\f`. If None then 440.0 is assumed.
    :type frequency: array or list of the appropriate size (or float if there is only one frequency parameter)
    :param ARD1: Auto Relevance Determination with respect to frequency.
        If equal to "False" one single frequency parameter :math:`\f` for
        each dimension is assumed, otherwise there is one lengthscale
        parameter per dimension.
    :type ARD1: Boolean
    :param active_dims: indices of dimensions which are used in the computation of the kernel
    :type active_dims: array or list of the appropriate size
    :param name: Name of the kernel for output
    :type String
    :param useGPU: whether or not use GPU
    :type Boolean
    """

    def __init__(self, input_dim, variance=1., frequencies=None, partial_count=None, ARD1=False,  active_dims=None, name='custom_mog_kernel', useGPU=False):
        super(CustomMogKernel, self).__init__(
            input_dim, active_dims, name, useGPU=useGPU)
        self.ARD1 = ARD1  # correspond to periods

        self.name = name

        # Configure for Auto Relevance Detection of frequencies
        if self.ARD1 == False:
            if frequencies is not None:
                frequencies = np.asarray(frequencies)
                assert frequencies.size == 1, "Only one frequency needed for non-ARD kernel"
            else:
                # We assume there is only 1 frequencies if frequencies is None and ARD1 is false
                frequencies = np.array([440.])
        else:
            if frequencies is not None:
                frequencies = np.asarray(frequencies)
                assert frequencies.size == input_dim, "unexpected number of frequencies"
            else:
                # We assume all frequencies are 440 Hz if ARD is true yet no frequencies are given
                frequencies = np.array([440.]*input_dim)

        # Configure the number of partials
        if partial_count is not None:
            partial_count = int(partial_count)
            assert type(
                partial_count) == int, "Partial count must be an integer"
        else:
            # We will automatically set the partial count to 10
            partial_count = 10

        self.variance = Param('variance', variance, Logexp())
        assert self.variance.size == 1, "Variance size must be one"
        self.frequencies = Param('frequencies', frequencies, Logexp())
        self.partial_count = Param('partial_count', partial_count, Logexp())

        self.link_parameters(
            self.variance,  self.frequencies, self.partial_count)

    def to_dict(self):
        """
        Convert the object into a json serializable dictionary.

        Note: It uses the private method _save_to_input_dict of the parent.

        :return dict: json serializable dictionary containing the needed information to instantiate the object
        """

        input_dict = super(CustomMogKernel, self)._save_to_input_dict()
        input_dict["class"] = "GPy.kern.CustomMogKernel"
        input_dict["variance"] = self.variance.values.tolist()
        input_dict["frequencies"] = self.frequencies.values.tolist()
        input_dict["partial count"] = self.partial_count.values.tolist()
        input_dict["ARD1"] = self.ARD1
        return input_dict

    def parameters_changed(self):
        """
        This functions deals as a callback for each optimization iteration.
        If one optimization step was successfull and the parameters
        this callback function will be called to be able to update any
        precomputations for the kernel.
        """

        pass

    def K(self, X, X2=None):
        """
        Compute the covariance matrix between X and X2 for the MoG spectral kernel matrix between two sets of vectors.

        X1 and X2 are arrays of shape (n1, d) and (n2, d), where n1 and n2 are the
        numbers of vectors and d is the dimension of each vector.

        M is the number of partials or harmonics for each note source.
        """
        if X2 is None:
            X2 = X

        n1 = X.shape[0]
        n2 = X2.shape[0]

        kernel_matrix = np.zeros((n1, n2))

        for i in range(n1):
            for j in range(n2):
                x1 = X[i, :]
                x2 = X2[j, :]

                cosine_series = 0
                for frequency in self.frequencies:
                    for m in range(len(self.partial_count)):
                        cosine_series += np.cos((m + 1) * 2 * np.pi *
                                                frequency * np.linalg.norm(x1 - x2))

                kernel_value = (1 / np.pi) * np.exp(-(self.variance / 2)
                                                    * np.linalg.norm(x1 - x2)**2) * cosine_series
                kernel_matrix[i, j] = kernel_value

        return kernel_matrix

    def Kdiag(self, X):
        """Compute the diagonal of the covariance matrix associated to X.
        i.e. this is the value for the covariance function when X - X1 = 0"""
        ret = np.empty(X.shape[0])
        ret[:] = (len(self.frequencies.shape) * self.partial_count) / np.pi
        return ret

    def dK_dX(self, X, X2, dimX):
        """
        Compute the derivative of K with respect to:
            dimension dimX of set X.
        """
        lengthscaleinv = (np.ones(X.shape[1])/(self.partial_count))[dimX]
        periodinv = (np.ones(X.shape[1])/(self.frequencies))[dimX]

        F = 0.5*np.pi*(lengthscaleinv**2)*periodinv  # multiplicative factor

        dist = X[:, None, dimX] - X2[None, :, dimX]
        base = np.pi*periodinv*dist

        return -F*np.sin(2*base)*self._clean_K(X, X2)

    def dK_dXdiag(self, X, dimX):
        """
        Compute the derivative of K with respect to:
            dimension dimX of set X.

        Returns only diagonal elements.
        """
        return np.zeros(X.shape[0])

    def dK_dX2(self, X, X2, dimX2):
        """
        Compute the derivative of K with respect to:
            dimension dimX2 of set X2.
        """
        return -self._clean_dK_dX(X, X2, dimX2)

    def dK_dX2diag(self, X, dimX2):
        """
        Compute the derivative of K with respect to:
            dimension dimX2 of set X2.

        Returns only diagonal elements.
        """
        return np.zeros(X.shape[0])

    def dK2_dXdX2(self, X, X2, dimX, dimX2):
        """
        Compute the second derivative of K with respect to:
            dimension dimX of set X, and
            dimension dimX2 of set X2.
        """
        lengthscaleinv = (np.ones(X.shape[1])/(self.partial_count))[dimX2]
        periodinv = (np.ones(X.shape[1])/(self.frequencies))[dimX2]

        F = 0.5*np.pi*(lengthscaleinv**2)*periodinv  # multiplicative factor

        dist = X[:, None, dimX2] - X2[None, :, dimX2]
        base = np.pi*periodinv*dist

        term = np.sin(2*base)*self._clean_dK_dX(X, X2, dimX)
        if dimX == dimX2:
            term += 2*np.pi*periodinv*np.cos(2*base)*self._clean_K(X, X2)
        return F*term

    def dK2_dXdX2diag(self, X, dimX, dimX2):
        """
        Compute the second derivative of K with respect to:
            dimension dimX of set X, and
            dimension dimX2 of set X2.

        Returns only diagonal elements.
        """
        if dimX == dimX2:
            lengthscaleinv = (np.ones(X.shape[1])/(self.partial_count))[dimX2]
            periodinv = (np.ones(X.shape[1])/(self.frequencies))[dimX2]
            return (np.pi**2)*(lengthscaleinv**2)*(periodinv**2)*self.variance*np.ones(X.shape[0])
        else:
            return np.zeros(X.shape[0])

    def dK2_dXdX(self, X, X2, dimX_0, dimX_1):
        """
        Compute the second derivative of K with respect to:
            dimension dimX_0 of set X, and
            dimension dimX_1 of set X.
        """
        return -self._clean_dK2_dXdX2(X, X2, dimX_0, dimX_1)

    def dK2_dXdXdiag(self, X, dimX_0, dimX_1):
        """
        Compute the second derivative of K with respect to:
            dimension dimX_0 of set X, and
            dimension dimX_1 of set X.

        Returns only diagonal elements.
        """
        return -self._clean_dK2_dXdX2diag(X, dimX_0, dimX_1)

    def dK3_dXdXdX2(self, X, X2, dimX_0, dimX_1, dimX2):
        """
        Compute the third derivative of K with respect to:
            dimension dimX_0 of set X,
            dimension dimX_1 of set X, and
            dimension dimX2 of set X2.
        """
        lengthscaleinv = (np.ones(X.shape[1])/(self.partial_count))[dimX2]
        periodinv = (np.ones(X.shape[1])/(self.frequencies))[dimX2]

        F = 0.5*np.pi*(lengthscaleinv**2)*periodinv  # multiplicative factor

        dist = X[:, None, dimX2] - X2[None, :, dimX2]
        base = np.pi*periodinv*dist

        term = np.sin(2*base)*self._clean_dK2_dXdX(X, X2, dimX_0, dimX_1)
        if dimX_0 == dimX2:
            term += 2*np.pi*periodinv * \
                np.cos(2*base)*self._clean_dK_dX(X, X2, dimX_1)
        if dimX_1 == dimX2:
            term += 2*np.pi*periodinv * \
                np.cos(2*base)*self._clean_dK_dX(X, X2, dimX_0)
        if dimX_0 == dimX_1 == dimX2:
            term -= 4*(np.pi**2)*(periodinv**2) * \
                np.sin(2*base)*self._clean_K(X, X2)
        return F*term

    def dK3_dXdXdX2diag(self, X, dimX_0, dimX_1, dimX2):
        """
        Compute the third derivative of K with respect to:
            dimension dimX_0 of set X,
            dimension dimX_1 of set X, and
            dimension dimX2 of set X2.

        Returns only diagonal elements of the covariance matrix.
        """
        return np.zeros(X.shape[0])

    def dK_dvariance(self, X, X2):
        """
        Compute the derivative of K with respect to variance.
        """
        return self._clean_K(X, X2)/self.variance

    def dK_dlengthscale(self, X, X2):
        """
        Compute the derivative(s) of K with respect to lengthscale(s).
        """
        lengthscaleinv = (np.ones(X.shape[1])/(self.partial_count))
        periodinv = (np.ones(X.shape[1])/(self.frequencies))

        dist = np.rollaxis(X[:, None, :] - X2[None, :, :], 2, 0)
        base = np.pi*periodinv[:, None, None]*dist

        K = self._clean_K(X, X2)

        if self.ARD2:
            g = []
            for diml in range(self.input_dim):
                g += [(lengthscaleinv[diml]**3) *
                      np.square(np.sin(base[diml]))*K]
        else:
            g = (lengthscaleinv[0]**3) * \
                np.sum(np.square(np.sin(base)), axis=0)*K
        return g

    def dK_dperiod(self, X, X2):
        """
        Compute the derivative(s) of K with respect to period(s).
        """
        lengthscaleinv = (np.ones(X.shape[1])/(self.partial_count))
        periodinv = (np.ones(X.shape[1])/(self.frequencies))

        dist = np.rollaxis(X[:, None, :] - X2[None, :, :], 2, 0)
        base = np.pi*periodinv[:, None, None]*dist

        K = self._clean_K(X, X2)

        if self.ARD1:
            g = []
            for diml in range(self.input_dim):
                g += [0.5*base[diml]*(lengthscaleinv[diml]**2)
                      * periodinv[diml]*np.sin(2*base[diml])*K]
        else:
            g = 0.5*periodinv[0]*np.sum(base*(lengthscaleinv**2)
                                        [:, None, None]*np.sin(2*base), axis=0)*K
        return g

    def dK2_dvariancedX(self, X, X2, dimX):
        """
        Compute the second derivative of K with respect to:
            variance, and
            dimension dimX of set X.
        """
        return self._clean_dK_dX(X, X2, dimX)/self.variance

    def dK2_dvariancedX2(self, X, X2, dimX2):
        """
        Compute the second derivative of K with respect to:
            variance, and
            dimension dimX2 of set X2.
        """
        return -self.dK2_dvariancedX(X, X2, dimX2)

    def dK2_dlengthscaledX(self, X, X2, dimX):
        """
        Compute the second derivative(s) of K with respect to:
            lengthscale(s), and
            dimension dimX of set X.
        """
        lengthscaleinv = (np.ones(X.shape[1])/(self.partial_count))[dimX]
        periodinv = (np.ones(X.shape[1])/(self.frequencies))[dimX]

        dist = X[:, None, dimX] - X2[None, :, dimX]
        base = np.pi*periodinv*dist

        F = 0.5*np.pi*(lengthscaleinv**2)*periodinv  # multiplicative factor

        K = self._clean_K(X, X2)
        dK_dl = self.dK_dlengthscale(X, X2)

        if self.ARD2:
            g = []
            for diml in range(self.input_dim):
                term = dK_dl[diml]
                if diml == dimX:
                    term -= 2*lengthscaleinv*K
                g += [-F*np.sin(2*base)*term]
        else:
            g = -F*np.sin(2*base)*(dK_dl - 2*lengthscaleinv*K)
        return g

    def dK2_dlengthscaledX2(self, X, X2, dimX2):
        """
        Compute the second derivative(s) of K with respect to:
            lengthscale(s), and
            dimension dimX2 of set X2.
        """
        dK2_dldX = self.dK2_dlengthscaledX(X, X2, dimX2)
        if self.ARD2:
            return [-1*g for g in dK2_dldX]
        else:
            return -1*dK2_dldX

    def dK2_dperioddX(self, X, X2, dimX):
        """
        Compute the second derivative(s) of K with respect to:
            period(s), and
            dimension dimX of set X.
        """
        lengthscaleinv = (np.ones(X.shape[1])/(self.partial_count))[dimX]
        periodinv = (np.ones(X.shape[1])/(self.frequencies))[dimX]

        dist = X[:, None, dimX] - X2[None, :, dimX]
        base = np.pi*periodinv*dist

        F = 0.5*np.pi*(lengthscaleinv**2)*periodinv  # multiplicative factor

        K = self._clean_K(X, X2)
        dK_dT = self.dK_dperiod(X, X2)

        if self.ARD1:
            g = []
            for dimT in range(self.input_dim):
                term = np.sin(2*base)*dK_dT[dimT]
                if dimT == dimX:
                    term -= periodinv*(np.sin(2*base)+2*base*np.cos(2*base))*K
                g += [-F*term]
        else:
            term = np.sin(2*base)*dK_dT
            term -= periodinv*(np.sin(2*base)+2*base*np.cos(2*base))*K
            g = -F*term
        return g

    def dK2_dperioddX2(self, X, X2, dimX2):
        """
        Compute the second derivative(s) of K with respect to:
            period(s), and
            dimension dimX2 of set X2.
        """
        dK2_dperioddX = self.dK2_dperioddX(X, X2, dimX2)
        if self.ARD1:
            return [-1*g for g in dK2_dperioddX]
        else:
            return -1*dK2_dperioddX

    def dK3_dvariancedXdX2(self, X, X2, dimX, dimX2):
        """
        Compute the third derivative of K with respect to:
            variance,
            dimension dimX of set X, and
            dimension dimX2 of set X2.
        """
        return self._clean_dK2_dXdX2(X, X2, dimX, dimX2)/self.variance

    def dK3_dlengthscaledXdX2(self, X, X2, dimX, dimX2):
        """
        Compute the third derivative(s) of K with respect to:
            lengthscale(s),
            dimension dimX of set X, and
            dimension dimX2 of set X2.
        """
        lengthscaleinv = (np.ones(X.shape[1])/(self.partial_count))[dimX2]
        periodinv = (np.ones(X.shape[1])/(self.frequencies))[dimX2]

        dist = X[:, None, dimX2] - X2[None, :, dimX2]
        base = np.pi*periodinv*dist

        F = 0.5*np.pi*(lengthscaleinv**2)*periodinv  # multiplicative factor

        dK2_dXdX2 = self._clean_dK2_dXdX2(X, X2, dimX, dimX2)
        dK_dl = self.dK_dlengthscale(X, X2)
        dK2_dldX = self.dK2_dlengthscaledX(X, X2, dimX)

        if self.ARD2:
            g = []
            for diml in range(self.input_dim):
                term = np.sin(2*base)*dK2_dldX[diml]
                if dimX == dimX2:
                    term += 2*np.pi*periodinv*np.cos(2*base)*dK_dl[diml]
                term *= F
                if diml == dimX2:
                    term -= 2*lengthscaleinv*dK2_dXdX2
                g += [term]
        else:
            term = np.sin(2*base)*dK2_dldX
            if dimX == dimX2:
                term += 2*np.pi*periodinv*np.cos(2*base)*dK_dl
            term *= F
            term -= 2*lengthscaleinv*dK2_dXdX2
            g = term
        return g

    def dK3_dperioddXdX2(self, X, X2, dimX, dimX2):
        """
        Compute the third derivative(s) of K with respect to:
            period(s),
            dimension dimX of set X, and
            dimension dimX2 of set X2.
        """
        lengthscaleinv = (np.ones(X.shape[1])/(self.partial_count))[dimX2]
        periodinv = (np.ones(X.shape[1])/(self.frequencies))[dimX2]

        dist = X[:, None, dimX2] - X2[None, :, dimX2]
        base = np.pi*periodinv*dist

        F = 0.5*np.pi*(lengthscaleinv**2)*periodinv  # multiplicative factor

        K = self._clean_K(X, X2)
        dK_dX = self._clean_dK_dX(X, X2, dimX)
        dK2_dXdX2 = self._clean_dK2_dXdX2(X, X2, dimX, dimX2)
        dK_dT = self.dK_dperiod(X, X2)
        dK2_dTdX = self.dK2_dperioddX(X, X2, dimX)

        if self.ARD1:
            g = []
            for dimT in range(self.input_dim):
                term = np.sin(2*base)*dK2_dTdX[dimT]
                if dimT == dimX2:
                    term -= 2*periodinv*np.cos(2*base)*base*dK_dX
                if dimX == dimX2:
                    term += 2*np.pi*periodinv*np.cos(2*base)*dK_dT[dimT]
                if dimX == dimX2 == dimT:
                    term += 2*np.pi*(periodinv**2) * \
                        (2*base*np.sin(2*base)-np.cos(2*base))*K
                term *= F
                if dimT == dimX2:
                    term -= periodinv*dK2_dXdX2
                g += [term]
        else:
            term = np.sin(2*base)*dK2_dTdX-2*periodinv * \
                base*np.cos(2*base)*dK_dX
            if dimX == dimX2:
                term += 2*np.pi*periodinv * \
                    (np.cos(2*base)*dK_dT+periodinv *
                     (2*base*np.sin(2*base)-np.cos(2*base))*K)
            g = F*term-periodinv*dK2_dXdX2
        return g

    def update_gradients_full(self, dL_dK, X, X2=None):
        """derivative of the covariance matrix with respect to the parameters."""
        if X2 is None:
            X2 = X

        base = np.pi * (X[:, None, :] - X2[None, :, :]) / self.frequencies

        sin_base = np.sin(base)
        exp_dist = np.exp(-0.5 * np.sum(np.square(sin_base /
                          self.partial_count), axis=-1))

        dwl = self.variance * (1.0/np.square(self.partial_count)) * \
            sin_base*np.cos(base) * (base / self.frequencies)

        dl = self.variance * np.square(sin_base) / \
            np.power(self.partial_count, 3)

        self.variance.gradient = np.sum(exp_dist * dL_dK)
        # target[0] += np.sum( exp_dist * dL_dK)

        if self.ARD1:  # different periods
            self.frequencies.gradient = (
                dwl * exp_dist[:, :, None] * dL_dK[:, :, None]).sum(0).sum(0)
        else:  # same period
            self.frequencies.gradient = np.sum(dwl.sum(-1) * exp_dist * dL_dK)

        if self.ARD2:  # different lengthscales
            self.partial_count.gradient = (
                dl * exp_dist[:, :, None] * dL_dK[:, :, None]).sum(0).sum(0)
        else:  # same lengthscales
            self.partial_count.gradient = np.sum(dl.sum(-1) * exp_dist * dL_dK)

    def update_gradients_direct(self, dL_dVar, dL_dPer, dL_dLen):
        self.variance.gradient = dL_dVar
        self.frequencies.gradient = dL_dPer
        self.partial_count.gradient = dL_dLen

    def reset_gradients(self):
        self.variance.gradient = 0.
        if not self.ARD1:
            self.frequencies.gradient = 0.
        else:
            self.frequencies.gradient = np.zeros(self.input_dim)
        if not self.ARD2:
            self.partial_count.gradient = 0.
        else:
            self.partial_count.gradient = np.zeros(self.input_dim)

    def update_gradients_diag(self, dL_dKdiag, X):
        """derivative of the diagonal of the covariance matrix with respect to the parameters."""
        self.variance.gradient = np.sum(dL_dKdiag)
        self.frequencies.gradient = 0
        self.partial_count.gradient = 0

    def dgradients(self, X, X2):
        g1 = self.dK_dvariance(X, X2)
        g2 = self.dK_dperiod(X, X2)
        g3 = self.dK_dlengthscale(X, X2)
        return [g1, g2, g3]

    def dgradients_dX(self, X, X2, dimX):
        g1 = self.dK2_dvariancedX(X, X2, dimX)
        g2 = self.dK2_dperioddX(X, X2, dimX)
        g3 = self.dK2_dlengthscaledX(X, X2, dimX)
        return [g1, g2, g3]

    def dgradients_dX2(self, X, X2, dimX2):
        g1 = self.dK2_dvariancedX2(X, X2, dimX2)
        g2 = self.dK2_dperioddX2(X, X2, dimX2)
        g3 = self.dK2_dlengthscaledX2(X, X2, dimX2)
        return [g1, g2, g3]

    def dgradients2_dXdX2(self, X, X2, dimX, dimX2):
        g1 = self.dK3_dvariancedXdX2(X, X2, dimX, dimX2)
        g2 = self.dK3_dperioddXdX2(X, X2, dimX, dimX2)
        g3 = self.dK3_dlengthscaledXdX2(X, X2, dimX, dimX2)
        return [g1, g2, g3]

    def gradients_X(self, dL_dK, X, X2=None):
        K = self.K(X, X2)
        if X2 is None:
            dL_dK = dL_dK+dL_dK.T
            X2 = X
        dX = -np.pi*((dL_dK*K)[:, :, None]*np.sin(2*np.pi/self.frequencies*(
            X[:, None, :] - X2[None, :, :]))/(2.*np.square(self.partial_count)*self.frequencies)).sum(1)
        return dX

    def gradients_X_diag(self, dL_dKdiag, X):
        return np.zeros(X.shape)

    def input_sensitivity(self, summarize=True):
        return self.variance*np.ones(self.input_dim)/self.partial_count**2
