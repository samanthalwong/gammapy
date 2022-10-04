# Licensed under a 3-clause BSD style license - see LICENSE.rst
# cython: language_level=3
import numpy as np

cimport numpy as np
cimport cython


cdef extern from "math.h":
    float log(float x)

cdef extern from "math.h":
    float sqrt(float x)

global TRUNCATION_VALUE
TRUNCATION_VALUE = 1e-25

@cython.cdivision(True)
@cython.boundscheck(False)
def cash_sum_cython(np.ndarray[np.float_t, ndim=1] counts,
                    np.ndarray[np.float_t, ndim=1] npred):
    """Summed cash fit statistics.

    Parameters
    ----------
    counts : `~numpy.ndarray`
        Counts array.
    npred : `~numpy.ndarray`
        Predicted counts array.
    """
    cdef np.float_t sum = 0
    cdef np.float_t npr, lognpr
    cdef unsigned int i, ni
    cdef np.float_t trunc = TRUNCATION_VALUE
    cdef np.float_t logtrunc = log(TRUNCATION_VALUE)

    ni = counts.shape[0]
    for i in range(ni):
        npr = npred[i]
        if npr > trunc:
            lognpr = log(npr)
        else:
            npr = trunc
            lognpr = logtrunc

        sum += npr
        if counts[i] > 0:
            sum -= counts[i] * lognpr

    return 2 * sum


@cython.cdivision(True)
@cython.boundscheck(False)
def f_cash_root_cython(np.float_t x, np.ndarray[np.float_t, ndim=1] counts,
                       np.ndarray[np.float_t, ndim=1] background,
                       np.ndarray[np.float_t, ndim=1] model):
    """Function to find root of. Described in Appendix A, Stewart (2009).

    Parameters
    ----------
    x : float
        Model amplitude.
    counts : `~numpy.ndarray`
        Count image slice, where model is defined.
    background : `~numpy.ndarray`
        Background image slice, where model is defined.
    model : `~numpy.ndarray`
        Source template (multiplied with exposure).
    """
    cdef np.float_t sum = 0
    cdef unsigned int i, ni
    ni = counts.shape[0]
    for i in range(ni):
        if model[i] > 0:
            if counts[i] > 0:
                sum += model[i] * (1 - counts[i] / (x * model[i] + background[i]))
            else:
                sum += model[i]

    # 2 is required to maintain the correct normalization of the
    # derivative of the likelihood function. It doesn't change the result of
    # the fit.
    return 2 * sum


@cython.cdivision(True)
@cython.boundscheck(False)
def norm_bounds_cython(np.ndarray[np.float_t, ndim=1] counts,
                            np.ndarray[np.float_t, ndim=1] background,
                            np.ndarray[np.float_t, ndim=1] model):
    """Compute bounds for the root of `_f_cash_root_cython`.

    Parameters
    ----------
    counts : `~numpy.ndarray`
        Counts image
    background : `~numpy.ndarray`
        Background image
    model : `~numpy.ndarray`
        Source template (multiplied with exposure).
    """
    cdef np.float_t s_model = 0, s_counts = 0, sn, sn_min = 1e14, c_min = 1
    cdef np.float_t b_min, b_max, sn_min_total = 1e14
    cdef unsigned int i, ni
    ni = counts.shape[0]
    for i in range(ni):
        if counts[i] > 0:
            s_counts += counts[i]
            if model[i] > 0:
                sn = background[i] / model[i]
                if sn < sn_min:
                    sn_min = sn
                    c_min = counts[i]
        if model[i] > 0:
            s_model += model[i]
            sn = background[i] / model[i]
            if sn < sn_min_total:
                sn_min_total = sn
    b_min = c_min / s_model - sn_min
    b_max = s_counts / s_model - sn_min
    return b_min, b_max, -sn_min_total

@cython.cdivision(True)
@cython.boundscheck(False)
def wstat_sum_cython(np.ndarray[np.float_t, ndim=1] n_on,
              np.ndarray[np.float_t, ndim=1] n_off,
              np.ndarray[np.float_t, ndim=1] alpha,
              np.ndarray[np.float_t, ndim=1] mu_sig):

    cdef np.float_t sum = 0
    cdef np.float_t C, D
    cdef np.float_t non, noff, alp, musig, mubkg, total
    cdef unsigned int i, ni
    cdef np.float_t trunc = TRUNCATION_VALUE

    ni = n_on.shape[0]
    for i in range(ni):
        alp = alpha[i]
        musig = mu_sig[i]
        non = n_on[i]
        noff = n_off[i]

        C = alp * (non + noff) - (1 + alp) * musig
        D = sqrt(C**2 + 4 * alp * (alp + 1) * noff * musig)
        mubkg = (C + D) / (2 * alp * (alp + 1))

        total = musig + (1 + alp) * mubkg
        if non > trunc:
            total += -non * (1+log(musig + alp * mubkg)- log(non))
        if noff > trunc:
            total += -noff *(1+log(mubkg)-log(noff))

        sum += total
    return 2*sum