from __future__ import print_function, division
from collections import namedtuple
import numpy as np
from agaro import output_utils
from bannock.utils import utils
from cannock.utils import utils as cutils


def vis_chi_k(dirnames, ax, label, c):
    m = output_utils.get_recent_model(dirnames[0])
    chis, ks, ks_err = utils.chi_ks(dirnames, t_steady=None)
    i_sort = np.argsort(chis)
    chis, ks, ks_err = chis[i_sort], ks[i_sort], ks_err[i_sort]
    D_rhos = cutils.get_D_rho(m.v_0, m.p_0, m.dim)
    mus = cutils.get_mu(chis, m.v_0, m.p_0, m.onesided_flag, m.L)
    mus_red = cutils.get_reduced_mu(mus, m.c_source, m.rho_0, m.c_sink, m.c_D)
    D_rhos_red = cutils.get_reduced_D_rho(D_rhos, m.c_D)
    ax.errorbar(mus_red / D_rhos_red, ks, yerr=ks_err, label=label, c=c)

ScanPlotSet = namedtuple('ScanPlotSet', ['dirname_paths', 'label', 'color'])
