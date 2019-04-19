import numpy as np
import matplotlib.pyplot as plt
from plot_tools import error_ellipse

def make_x(star):
    """
    returns a vector of x = [parallax, pmra, pmdec]
    """
    names = ['parallax', 'pmra', 'pmdec']
    try:
        return star.loc[names].values.astype('f')
    except:
        return star[names].values[0].astype('f')

def make_xerr(star):
    """
    returns a vector of xerr = [parallax_error, pmra_error, pmdec_error]
    """
    err_names = ['parallax_error', 'pmra_error', 'pmdec_error']
    try:
        return star.loc[err_names].values.astype('f')
    except:
        return star[err_names].values[0].astype('f')

def ppm_check(star1, star2, sigma=5.):
    """
    Returns True if the differences between parallax, pmra, and pmdec are all below
    the sigma threshold.
    """
    x1 = make_x(star1)
    x2 = make_x(star2)
    if np.any(np.isnan([x1,x2])):
        return False
    xerr1 = make_xerr(star1)
    xerr2 = make_xerr(star2)
    if np.any(np.isnan([xerr1, xerr2])):
        return False
    if np.any(np.abs(x1 - x2)/np.sqrt(xerr1**2 + xerr2**2) >= sigma):
        return False
    return True

def make_cov(star):
    """
    returns covariance matrix C corresponding to x
    """
    names = ['parallax', 'pmra', 'pmdec']
    C = np.diag(make_xerr(star)**2)
    for i, name1 in enumerate(names):
        for j, name2 in enumerate(names):
            if j >= i:
                continue
            try:
                corr = star.loc["{0}_{1}_corr".format(name2, name1)]
            except:
                corr = star["{0}_{1}_corr".format(name2, name1)].values[0]
            C[i, j] = corr * np.sqrt(C[i, i] * C[j, j])
            C[j, i] = C[i, j]
    return C

def star_is_good(star):
    """
    determine whether star meets the following requirements:
     - has 5-parameter solution
     - has >8 visibility periods used in solution
     - low astrometric excess noise (as defined in Gaia Collaboration (2018) H-R diagram paper)
     - has a significantly non-zero proper motion in either RA or Dec
    returns boolean
    """
    plx_check = np.isfinite(star.loc['parallax'])
    if not plx_check:
        return False
    vis_periods_check = star.loc['visibility_periods_used'] > 8
    if not vis_periods_check:
        return False
    pm_check = (star.loc['pmra']/star.loc['pmra_error'] >= 3.) \
                or (star.loc['pmdec']/star.loc['pmdec_error'] >= 3.)
    if not pm_check:
        return False
    chi2 = star.loc['astrometric_chi2_al']
    nu_prime = star.loc['astrometric_n_good_obs_al']
    mg = star.loc['phot_g_mean_mag']
    plx_noise_check = np.sqrt(chi2/(nu_prime - 5.)) < 1.2*max([1., np.exp(-0.2*(mg - 19.5))])   
    if not plx_noise_check:
        return False
    return True

def chisq(star1, star2):
    """
    calculates chisquared for two stars based on their parallax and 2D proper motions
    """
    deltax = make_x(star1) - make_x(star2)
    cplusc = make_cov(star1) + make_cov(star2)
    return np.dot(deltax, np.linalg.solve(cplusc, deltax))
    
def calc_chisq_for_pair(m, primary):
    if star_is_good(m) & ppm_check(primary, m):
        return chisq(primary, m)
    else:
        return -1
    
def calc_chisq_nonzero(star):
    """
    Chisquared-like metric to diagnose how different from zero the proper motions
    Does NOT take parallax into account
    """
    x = make_x(star)[1:]
    cov = make_cov(star)[1:,1:]
    return np.dot(x, np.linalg.solve(cov, x))
    
def plot_xs(star1, star2, sigma=1):
    fs = 12
    x1 = make_x(star1)
    cov1 = make_cov(star1)
    x2 = make_x(star2)
    cov2 = make_cov(star2)
    fig = plt.figure(figsize=(12,4))
    ax1 = fig.add_subplot(131)
    error_ellipse(ax1, x1[0], x1[1], cov1[:2,:2], ec='red', sigma=sigma)
    error_ellipse(ax1, x2[0], x2[1], cov2[:2,:2], ec='blue', sigma=sigma)
    ax1.set_xlim([min([x1[0], x2[0]]) - 5., max([x1[0], x2[0]]) + 5.])
    ax1.set_ylim([min([x1[1], x2[1]]) - 5., max([x1[1], x2[1]]) + 5.])
    ax1.set_xlabel('Parallax (mas)', fontsize=fs)
    ax1.set_ylabel('PM RA (mas yr$^{-1}$)', fontsize=fs)

    ax2 = fig.add_subplot(133)
    error_ellipse(ax2, x1[1], x1[2], cov1[1:,1:], ec='red', sigma=sigma)
    error_ellipse(ax2, x2[1], x2[2], cov2[1:,1:], ec='blue', sigma=sigma)
    ax2.set_xlim([min([x1[1], x2[1]]) - 5., max([x1[1], x2[1]]) + 5.])
    ax2.set_ylim([min([x1[2], x2[2]]) - 5., max([x1[2], x2[2]]) + 5.])
    ax2.set_xlabel('PM RA (mas yr$^{-1}$)', fontsize=fs)
    ax2.set_ylabel('PM Dec (mas yr$^{-1}$)', fontsize=fs)

    ax3 = fig.add_subplot(132)
    c1 = np.delete(np.delete(cov1, 1, axis=0), 1, axis=1)
    c2 = np.delete(np.delete(cov2, 1, axis=0), 1, axis=1)
    error_ellipse(ax3, x1[0], x1[2], c1, ec='red', sigma=sigma)
    error_ellipse(ax3, x2[0], x2[2], c2, ec='blue', sigma=sigma)
    ax3.set_xlim([min([x1[0], x2[0]]) - 5., max([x1[0], x2[0]]) + 5.])
    ax3.set_ylim([min([x1[2], x2[2]]) - 5., max([x1[2], x2[2]]) + 5.])
    ax3.set_xlabel('Parallax (mas)', fontsize=fs)
    ax3.set_ylabel('PM Dec (mas yr$^{-1}$)', fontsize=fs)

    fig.subplots_adjust(wspace = 0.5)
    #fig.text(0.5, 0.95, 'match #{0}'.format(i), horizontalalignment='center',
    #         transform=ax3.transAxes, fontsize=fs+2)