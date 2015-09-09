from __future__ import print_function, division
from itertools import product
import numpy as np
from agaro.run_utils import run_model, run_kwarg_scan
from bannock import model, walls

L = 2480.0
dx = 40.0

trap_d = 40.0
trap_w = 280.0
trap_s = 120.0

maze_d = 40.0
maze_seed = 1

seed = 1
dt = 0.1
v_0 = 20.0
p_0 = 1.0
origin_flag = False
vicsek_R = 0.0
c_D = 1000.0
c_sink = 0.01
c_source = 1.0

D_rot = 0.2
force_mu = 0.0

default_model_1d_kwargs = {
    'seed': seed,
    'dt': dt,
    'v_0': v_0,
    'p_0': p_0,
    'origin_flag': False,
    'vicsek_R': 0.0,
    'L': L,
    'dx': dx,
    'c_D': c_D,
    'c_sink': c_sink,
    'c_source': c_source,
}

default_model_2d_kwargs = dict(default_model_1d_kwargs,
                               D_rot=D_rot, force_mu=force_mu)


def run_test_2d():
    trap_n = 1
    extra_model_kwargs = {
        'rho_0': 2e-4,
        'onesided_flag': True,
        'chi': 0.0,
        'walls': walls.Traps(L, dx, trap_n, trap_d, trap_w, trap_s),
        'origin_flag': True,
    }
    model_kwargs = dict(default_model_2d_kwargs, **extra_model_kwargs)
    m = model.Model2D(**model_kwargs)

    t_output_every = 20.0
    t_upto = 100.0
    output_dir = 'test_2d'
    force_resume = None

    run_model(t_output_every, output_dir, m, force_resume, t_upto=t_upto)


def run_test_1d():
    extra_model_kwargs = {
        'rho_0': 0.1,
        'onesided_flag': True,
        'chi': 0.0,
        'origin_flag': True,
    }
    model_kwargs = dict(default_model_1d_kwargs, **extra_model_kwargs)
    m = model.Model1D(**model_kwargs)

    t_output_every = 20.0
    t_upto = 100.0
    output_dir = 'test_1d'
    force_resume = None

    run_model(t_output_every, output_dir, m, force_resume, t_upto=t_upto)


def run_chi_scan_1d():
    extra_model_kwargs = {
        'rho_0': 0.5,
        'onesided_flag': True,
        'origin_flag': True,
    }
    model_kwargs = dict(default_model_1d_kwargs, **extra_model_kwargs)

    t_output_every = 1000.0
    t_upto = 4e4
    chis = np.linspace(0.0, 6.0, 28)
    force_resume = True
    parallel = True

    model_kwarg_sets = []

    origin_flags = [True, False]
    onesided_flags = [True, False]
    for origin_flag, onesided_flag, chi in product(origin_flags,
                                                   onesided_flags, chis):
        model_kwargs_cur = model_kwargs.copy()
        model_kwargs_cur['onesided_flag'] = onesided_flag
        model_kwargs_cur['origin_flag'] = origin_flag
        model_kwargs_cur['chi'] = chi
        model_kwarg_sets.append(model_kwargs_cur)
    run_kwarg_scan(model.Model1D, model_kwarg_sets,
                   t_output_every, t_upto, force_resume, parallel)


def run_cannock_1d():
    extra_model_kwargs = {
        'rho_0': 1.0,
        'onesided_flag': True,
        'chi': 1.5,
        'origin_flag': True,
    }
    model_kwargs = dict(default_model_1d_kwargs, **extra_model_kwargs)
    m = model.Model1D(**model_kwargs)

    t_output_every = 50.0
    t_upto = 1e4
    output_dir = '/Users/ewj/Desktop/cannock/agent_data/{}'.format(m)
    force_resume = None

    run_model(t_output_every, output_dir, m, force_resume, t_upto=t_upto)


def run_chi_scan_2d_uniform():
    extra_model_kwargs = {
        'rho_0': 1e-3,
        'walls': walls.Walls(L, dim=2, dx=dx),
    }
    model_kwargs = dict(default_model_2d_kwargs, **extra_model_kwargs)

    t_output_every = 1000.0
    t_upto = 8e4
    chis = np.linspace(0.0, 800.0, 22)
    force_resume = True
    parallel = True

    model_kwarg_sets = []

    origin_flags = [True, False]
    onesided_flags = [True, False]
    for origin_flag, onesided_flag, chi in product(origin_flags,
                                                   onesided_flags, chis):
        model_kwargs_cur = model_kwargs.copy()
        model_kwargs_cur['onesided_flag'] = onesided_flag
        model_kwargs_cur['origin_flag'] = origin_flag
        model_kwargs_cur['chi'] = chi
        model_kwarg_sets.append(model_kwargs_cur)
    run_kwarg_scan(model.Model2D, model_kwarg_sets,
                   t_output_every, t_upto, force_resume, parallel)


def run_trap_no_chi():
    trap_n = 1
    extra_model_kwargs = {
        'rho_0': 1e-1,
        'onesided_flag': False,
        'chi': 0.0,
        'walls': walls.Traps(L, dx, trap_n, trap_d, trap_w, trap_s),
        'origin_flag': False,
        'p_0': 1.0,
        'c_source': 0.0,
    }
    model_kwargs = dict(default_model_2d_kwargs, **extra_model_kwargs)
    m = model.Model2D(**model_kwargs)

    t_output_every = 100.0
    t_upto = 4e4
    output_dir = None
    force_resume = None

    run_model(t_output_every, output_dir, m, force_resume, t_upto=t_upto)


def run_chi_scan_trap():
    extra_model_kwargs = {
        'rho_0': 1e-3,
        'onesided_flag': True,
        'walls': walls.Traps(L, dx=dx, n=1, d=trap_d, w=trap_w, s=trap_s),
    }
    model_kwargs = dict(default_model_2d_kwargs, **extra_model_kwargs)

    t_output_every = 1000.0
    t_upto = 8e4
    chis = np.linspace(0.0, 800.0, 22)
    force_resume = True
    parallel = True

    model_kwarg_sets = []

    origin_flags = [True, False]
    for origin_flag, chi in product(origin_flags, chis):
        model_kwargs_cur = model_kwargs.copy()
        model_kwargs_cur['origin_flag'] = origin_flag
        model_kwargs_cur['chi'] = chi
        model_kwarg_sets.append(model_kwargs_cur)
    run_kwarg_scan(model.Model2D, model_kwarg_sets,
                   t_output_every, t_upto, force_resume, parallel)


def run_chi_scan_traps():
    extra_model_kwargs = {
        'rho_0': 1e-3,
        'onesided_flag': True,
        'origin_flag': False,
        'walls': walls.Traps(L, dx=dx, n=5, d=trap_d, w=trap_w, s=trap_s),
    }
    model_kwargs = dict(default_model_2d_kwargs, **extra_model_kwargs)

    t_output_every = 1000.0
    t_upto = 8e4
    chis = np.linspace(0.0, 800.0, 22)
    force_resume = True
    parallel = True

    model_kwarg_sets = []

    for chi in chis:
        model_kwargs_cur = model_kwargs.copy()
        model_kwargs_cur['chi'] = chi
        model_kwarg_sets.append(model_kwargs_cur)
    run_kwarg_scan(model.Model2D, model_kwarg_sets,
                   t_output_every, t_upto, force_resume, parallel)


def run_trap_s_scan():
    extra_model_kwargs = {
        # Quarter dt to keep diffusion stable.
        'dt': 0.025,
        # Halve dx to let us do finer increments in `s`.
        'dx': 20.0,
        'origin_flag': True,
        'rho_0': 1e-3,
        'chi': None,
        'onesided_flag': True,
    }
    model_kwargs = dict(default_model_2d_kwargs, **extra_model_kwargs)

    t_output_every = 2000.0
    t_upto = 16e4
    chis = np.linspace(200.0, 600.0, 22)
    force_resume = True
    parallel = True

    model_kwarg_sets = []
    for s in [20.0, 60.0, 100.0, 140.0, 180.0]:
        walls_cur = walls.Traps(L, dx, n=1, d=trap_d, w=trap_w, s=trap_s)
        for chi in chis:
            model_kwargs_cur = model_kwargs.copy()
            model_kwargs_cur['walls'] = walls_cur
            model_kwargs_cur['chis'] = chi
        model_kwarg_sets.append(model_kwargs_cur)
    run_kwarg_scan(model.Model2D, model_kwarg_sets,
                   t_output_every, t_upto, force_resume, parallel)


def run_chi_scan_maze():
    extra_model_kwargs = {
        'rho_0': 1e-3,
        'onesided_flag': True,
        'walls': walls.Maze(L, dim=2, dx=dx, d=maze_d, seed=maze_seed)
    }
    model_kwargs = dict(default_model_2d_kwargs, **extra_model_kwargs)

    t_output_every = 1000.0
    t_upto = 4e4
    chis = np.linspace(0.0, 800.0, 22)
    force_resume = True
    parallel = True

    model_kwarg_sets = []

    origin_flags = [True, False]
    for origin_flag, chi in product(origin_flags, chis):
        model_kwargs_cur = model_kwargs.copy()
        model_kwargs_cur['origin_flag'] = origin_flag
        model_kwargs_cur['chi'] = chi
        model_kwarg_sets.append(model_kwargs_cur)
    run_kwarg_scan(model.Model2D, model_kwarg_sets,
                   t_output_every, t_upto, force_resume, parallel)
