from ase.units import Rydberg, Bohr
import numpy as np
import os
import warnings


def occupations(calc, val):
    """"""
    assert isinstance(val, str)
    values = ['smearing', 'tetrahedra', 'tetrahedra_lin',
              'tetrahedra_opt', 'fixed', 'from_input']

    assert val in values


def degauss(calc, val):
    """"""
    assert isinstance(val, (float, int))
    val /= Rydberg

    # Without smearing, we enforce fixed occupation.
    if abs(val) <= 1e-13:
        calc.params['occupations'] = 'fixed'
        moments = calc.atoms.get_initial_magnetic_moments()
        calc.params['tot_magnetization'] = moments.sum()
        warnings.warn("'degauss' is zero. Enforcing fixed "
                      "overall magnetic moment")

    return val


def lspinorb(calc, val):
    """"""
    assert isinstance(val, bool)
    assert calc.get_param('noncolin')


def nspin(calc, val):
    """"""
    assert val in [2, 4]
    assert calc.get_param('occupations')
    assert calc.get_param('smearing')
    assert calc.get_param('degauss')
    if val == 4:
        assert calc.get_param('noncolin')

    moments = calc.atoms.get_initial_magnetic_moments()
    assert moments.any()


def noncolin(calc, val):
    """"""
    assert isinstance(val, bool)
    calc.params['nspin'] = 4
    nspin(calc, 4)


def tot_magnetization(calc, val):
    """"""
    assert isinstance(val, int)
    nspin(calc, calc.get_param('nspin'))


def ion_dynamics(calc, val):
    """"""
    assert isinstance(val, str)

    calculator = calc.get_param('calculation')
    calculation(calc, calculator)

    if calculator == 'relax':
        if calc.atoms.constraints is not None:
            values = ['damp']
        else:
            values = ['bfgs', 'damp']

        assert val in values

    elif calculator == 'md':
        if calc.atoms.constraints is not None:
            values = ['verlet']
        else:
            values = ['verlet', 'langevin', 'langevin-smc']

        assert val in values

    elif calculator == 'vc-relax':
        values = ['bfgs', 'damp']
        assert val in values

    elif calculator == 'vc-md':
        values = ['beeman']
        assert val in values

    else:
        raise ValueError("ion_dynamics requires calculator to be "
                         "'relax', 'md', 'vc-relax', or 'vc-md'")


def nbnd(calc, val):
    """If nbands is negative, assign additional bands."""
    assert isinstance(val, int)

    if val < 0:
        nvalence, nel = calc.get_nvalence()

        if calc.get_param('noncolin'):
            val = nvalence.sum() - val * 2
        else:
            val = int(nvalence.sum() / 2) - val

    return val


def etot_conv_thr(calc, val):
    """"""
    assert isinstance(val, (float, int))
    val /= Rydberg

    return val


def forc_conv_thr(calc, val):
    """"""
    assert isinstance(val, (float, int))
    val /= Rydberg / Bohr

    return val


def ecutwfc(calc, val):
    """"""
    assert isinstance(val, (float, int))
    val /= Rydberg

    return val


def ecutrho(calc, val):
    """"""
    assert isinstance(val, (float, int))
    val /= Rydberg

    return val


def conv_thr(calc, val):
    """"""
    assert isinstance(val, (float, int))
    val /= Rydberg

    return val


def ecutfock(calc, val):
    """"""
    assert isinstance(val, (float, int))
    val /= Rydberg

    return val


def calculation(calc, val):
    """"""
    assert isinstance(val, str)

    values = ['scf', 'nscf', 'bands', 'relax', 'md', 'vc-relax', 'vc-md']
    assert val in values


def prefix(calc, val):
    """"""
    assert isinstance(val, str)
    warnings.warn("For directory consistency, 'prefix' is ignored")


def outdir(calc, val):
    """"""
    assert isinstance(val, str)


def disk_io(calc, val):
    """"""
    assert isinstance(val, str)

    if isinstance(val, str):
        values = ['high', 'medium', 'low', 'none']
        assert val in values


def kpts(calc, val):
    """"""
    assert isinstance(val, (tuple, list, np.ndarray))
    assert len(val) == 3


def input_dft(calc, val):
    """"""
    assert isinstance(val, str)


def mixing_beta(calc, val):
    """"""
    assert isinstance(val, float)
    assert (val > 0) and (val < 1)


def nosym(calc, val):
    """"""
    assert isinstance(val, bool)


variables = {
    # CONTROL
    'calculation': 'scf',
    'title': None,
    'verbosity': None,
    'restart_mode': None,
    'wf_collect': None,
    'nstep': None,
    'iprint': None,
    'tstress': True,
    'tprnfor': True,
    'dt': None,
    'outdir': '.',
    'wfcdir': None,
    'prefix': 'calc',
    'lkpoint_dir': None,
    'max_seconds': None,
    'etot_conv_thr': 0.0,
    'forc_conv_thr': 0.05 / (Rydberg / Bohr),
    'disk_io': None,
    'pseudo_dir': os.environ['ESP_PSP_PATH'],
    'tefield': None,
    'dipfield': None,
    'lelfield': None,
    'nberrycyc': None,
    'lorbm': None,
    'lberry': None,
    'gdir': None,
    'nppstr': None,
    'lfcpopt': None,
    'gate': None,
    # SYSTEM
    'ibrav': 0,
    'celldm': None,
    'A': None,
    'B': None,
    'C': None,
    'cosAB': None,
    'cosAC': None,
    'cosBC': None,
    'nat': None,
    'ntyp': None,
    'nbnd': None,
    'tot_charge': None,
    'starting_charge': None,
    'tot_magnetization': None,
    'starting_magnetization': None,
    'ecutwfc': None,
    'ecutrho': None,
    'ecutfock': None,
    'nr1': None,
    'nr2': None,
    'nr3': None,
    'nr1s': None,
    'nr2s': None,
    'nr3s': None,
    'nosym': None,
    'nosym_evc': None,
    'noinv': None,
    'no_t_rev': None,
    'force_symmorphic': None,
    'use_all_frac': None,
    'occupations': 'smearing',
    'one_atom_occupations': None,
    'starting_spin_angle': None,
    'degauss': 0.1 / Rydberg,
    'smearing': 'fd',
    'nspin': None,
    'noncolin': None,
    'ecfixed': None,
    'qcutz': None,
    'q2sigma': None,
    'input_dft': None,
    'exx_fraction': None,
    'screening_parameter': None,
    'exxdiv_treatment': None,
    'x_gamma_extrapolation': None,
    'ecutvcut': None,
    'nqx1': None,
    'nqx2': None,
    'nqx3': None,
    'lda_plus_u': None,
    'lda_plus_u_kind': None,
    'Hubbard_U': None,
    'Hubbard_J0': None,
    'Hubbard_alpha': None,
    'Hubbard_beta': None,
    'Hubbard_J': None,
    'starting_ns_eigenvalue': None,
    'U_projection_type': None,
    'edir': None,
    'emaxpos': None,
    'eopreg': None,
    'eamp': None,
    'angle1': None,
    'angle2': None,
    'constrained_magnetization': None,
    'fixed_magnetization': None,
    'lambda': None,
    'report': None,
    'lspinorb': None,
    'assume_isolated': None,
    'esm_bc': None,
    'esm_w': None,
    'esm_efield': None,
    'esm_nfit': None,
    'fcp_mu': None,
    'vdw_corr': None,
    'london': None,
    'london_s6': None,
    'london_c6': None,
    'london_rvdw': None,
    'london_rcut': None,
    'ts_vdw_econv_thr': None,
    'ts_vdw_isolated': None,
    'xdm': None,
    'xdm_a1': None,
    'xdm_a2': None,
    'space_group': None,
    'uniqueb': None,
    'origin_choice': None,
    'rhombohedral': None,
    'zgate': None,
    'realxz': None,
    'block': None,
    'block_1': None,
    'block_2': None,
    'block_height': None,
    # ELECTRONS
    'electron_maxstep': None,
    'scf_must_converge': None,
    'conv_thr': None,
    'adaptive_thr': None,
    'conv_thr_init': None,
    'conv_thr_multi': None,
    'mixing_mode': None,
    'mixing_beta': None,
    'mixing_ndim': None,
    'mixing_fixed_ns': None,
    'diagonalization': None,
    'ortho_para': None,  # OBSOLETE
    'diago_thr_init': None,
    'diago_cg_maxiter': None,
    'diago_david_ndim': None,
    'diago_full_acc': None,
    'efield': None,
    'efield_cart': None,
    'efield_phase': None,
    'startingpot': None,
    'startingwfc': None,
    'tqr': None,
    # IONS
    'ion_dynamics': None,
    'ion_positions': None,
    'pot_extrapolation': None,
    'wfc_extrapolation': None,
    'remove_rigid_rot': None,
    'ion_temperature': None,
    'tempw': None,
    'tolp': None,
    'delta_t': None,
    'nraise': None,
    'refold_pos': None,
    'upscale': None,
    'bfgs_ndim': None,
    'trust_radius_max': None,
    'trust_radius_min': None,
    'trust_radius_ini': None,
    'w_1': None,
    'w_2': None,
    # CELL
    'cell_dynamics': None,
    'press': None,
    'wmass': None,
    'cell_factor': None,
    'press_conv_thr': None,
    'cell_dofree': None,
}
