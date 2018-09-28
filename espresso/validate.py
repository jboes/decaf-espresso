from . import utils
from ase.io.espresso import KEYS
from ase.units import create_units
import numpy as np
import os
import warnings
import six
KEYS['system'] += ['ensemble_energies', 'print_ensemble_energies']
Rydberg = create_units('2006')['Rydberg']
Bohr = create_units('2006')['Bohr']

variables = {
    # CONTROL
    'outdir': '.',
    'prefix': 'calc',
    'etot_conv_thr': 1e12,
    'forc_conv_thr': 0.05 / (Rydberg / Bohr),
    'pseudo_dir': os.environ['ESP_PSP_PATH'],
    'occupations': 'smearing',
    'smearing': 'fd',
    'ibrav': 0,
    'degauss': 0.1 / Rydberg}

projwfc_vars = {
    'Emin': -20,
    'Emax': 20,
    'DeltaE': 0.01}


def edir(calc, val):
    """https://www.quantum-espresso.org/Doc/INPUT_PW.html#edir
    """
    assert isinstance(val, int)
    assert val in [1, 2, 3]


def emaxpos(calc, val):
    """https://www.quantum-espresso.org/Doc/INPUT_PW.html#emaxpos
    """
    assert isinstance(val, float)
    assert (val >= 0) and (val <= 1)


def eopreg(calc, val):
    """https://www.quantum-espresso.org/Doc/INPUT_PW.html#eopreg
    """
    assert isinstance(val, float)
    assert (val >= 0) and (val <= 1)


def eamp(calc, val):
    """https://www.quantum-espresso.org/Doc/INPUT_PW.html#eamp
    """
    assert isinstance(val, float)


def tefield(calc, val):
    """By default, the amplitude is set to zero, but if it is manually
    specified with an emaxpos, attempt to automatically assign emaxpos.

    https://www.quantum-espresso.org/Doc/INPUT_PW.html#tefield
    """
    assert isinstance(val, bool)
    if not calc.get_param('edir'):
        calc.parameters['edir'] = 3

    if not calc.get_param('eamp'):
        calc.parameters['eamp'] = 0.0

    if not calc.get_param('emaxpos'):
        calc.parameters['emaxpos'] = utils.get_max_empty_space(calc.atoms)


def dipfield(calc, val):
    """https://www.quantum-espresso.org/Doc/INPUT_PW.html#dipfield
    """
    assert isinstance(val, bool)
    if not calc.get_param('tefield'):
        calc.parameters['tefield'] = True
        tefield(calc, val)


def tstress(calc, val):
    """https://www.quantum-espresso.org/Doc/INPUT_PW.html#tstress
    """
    assert isinstance(val, bool)
    if not calc.get_param('tprnfor'):
        calc.parameters['tprnfor'] = True


def tprnfor(calc, val):
    """https://www.quantum-espresso.org/Doc/INPUT_PW.html#tprnfor
    """
    assert isinstance(val, bool)


def occupations(calc, val):
    """https://www.quantum-espresso.org/Doc/INPUT_PW.html#occupations
    """
    assert isinstance(val, six.string_types)
    values = ['smearing', 'tetrahedra', 'tetrahedra_lin',
              'tetrahedra_opt', 'fixed', 'from_input']

    assert val in values


def degauss(calc, val):
    """https://www.quantum-espresso.org/Doc/INPUT_PW.html#degauss
    """
    assert isinstance(val, (float, int))
    val /= Rydberg

    # Without smearing, we enforce fixed occupation.
    if abs(val) <= 1e-13:
        calc.parameters['occupations'] = 'fixed'
        moments = calc.atoms.get_initial_magnetic_moments()
        calc.parameters['tot_magnetization'] = moments.sum()
        calc.parameters['nspin'] = 2
        warnings.warn("'degauss' is zero. Enforcing fixed "
                      "overall magnetic moment")

    return val


def lspinorb(calc, val):
    """https://www.quantum-espresso.org/Doc/INPUT_PW.html#lspinorb
    """
    assert isinstance(val, bool)
    assert calc.get_param('noncolin')


def nspin(calc, val):
    """https://www.quantum-espresso.org/Doc/INPUT_PW.html#nspin
    """
    assert val in [2, 4]
    assert calc.get_param('occupations')
    assert calc.get_param('smearing')
    assert calc.get_param('degauss')
    if val == 4:
        assert calc.get_param('noncolin')

    moments = calc.atoms.get_initial_magnetic_moments()
    assert moments.any()


def noncolin(calc, val):
    """https://www.quantum-espresso.org/Doc/INPUT_PW.html#noncolin
    """
    assert isinstance(val, bool)
    calc.parameters['nspin'] = 4
    nspin(calc, 4)


def tot_magnetization(calc, val):
    """https://www.quantum-espresso.org/Doc/INPUT_PW.html#tot_magnetization
    """
    assert isinstance(val, int)
    nspin(calc, calc.get_param('nspin'))


def ion_dynamics(calc, val):
    """https://www.quantum-espresso.org/Doc/INPUT_PW.html#ion_dynamics
    """
    assert isinstance(val, six.string_types)

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
    """If nbnd is negative, assign additional bands of that quantity.

    https://www.quantum-espresso.org/Doc/INPUT_PW.html#nbnd
    """
    assert isinstance(val, int)

    if val < 0 and calc.atoms:
        nvalence, nel = calc.get_nvalence()

        if calc.get_param('noncolin'):
            val = nvalence.sum() - val * 2
        else:
            val = int(nvalence.sum() / 2) - val

    return val


def etot_conv_thr(calc, val):
    """https://www.quantum-espresso.org/Doc/INPUT_PW.html#etot_conv_thr
    """
    assert isinstance(val, (float, int))
    val /= Rydberg

    return val


def forc_conv_thr(calc, val):
    """https://www.quantum-espresso.org/Doc/INPUT_PW.html#forc_conv_thr
    """
    assert isinstance(val, (float, int))
    val /= Rydberg / Bohr

    return val


def ecutwfc(calc, val):
    """https://www.quantum-espresso.org/Doc/INPUT_PW.html#ecutwfc
    """
    assert isinstance(val, (float, int))
    val /= Rydberg

    return val


def ecutrho(calc, val):
    """https://www.quantum-espresso.org/Doc/INPUT_PW.html#ecutrho
    """
    assert isinstance(val, (float, int))
    val /= Rydberg

    return val


def conv_thr(calc, val):
    """https://www.quantum-espresso.org/Doc/INPUT_PW.html#conv_thr
    """
    assert isinstance(val, (float, int))
    val /= Rydberg

    return val


def ecutfock(calc, val):
    """https://www.quantum-espresso.org/Doc/INPUT_PW.html#ecutfock
    """
    assert isinstance(val, (float, int))
    val /= Rydberg

    return val


def calculation(calc, val):
    """https://www.quantum-espresso.org/Doc/INPUT_PW.html#calculation
    """
    assert isinstance(val, six.string_types)

    values = ['scf', 'nscf', 'bands', 'relax', 'md', 'vc-relax', 'vc-md']
    assert val in values


def prefix(calc, val):
    """https://www.quantum-espresso.org/Doc/INPUT_PW.html#prefix
    """
    assert isinstance(val, six.string_types)
    warnings.warn("For directory consistency, 'prefix' is ignored")


def outdir(calc, val):
    """https://www.quantum-espresso.org/Doc/INPUT_PW.html#outdir
    """
    assert isinstance(val, six.string_types)


def disk_io(calc, val):
    """https://www.quantum-espresso.org/Doc/INPUT_PW.html#disk_io
    """
    assert isinstance(val, six.string_types)

    if isinstance(val, str):
        values = ['high', 'medium', 'low', 'none']
        assert val in values


def kpts(calc, val):
    """Test k-points to be 'gamma' or list_like of 3 values.
    Only automatic assignment is currently supported.

    https://www.quantum-espresso.org/Doc/INPUT_PW.html#k_points
    """
    if val == 'gamma':
        return
    assert isinstance(val, (tuple, list, np.ndarray))
    assert len(val) == 3


def input_dft(calc, val):
    """https://www.quantum-espresso.org/Doc/INPUT_PW.html#input_dft
    """
    assert isinstance(val, six.string_types)


def mixing_beta(calc, val):
    """https://www.quantum-espresso.org/Doc/INPUT_PW.html#mixing_beta
    """
    assert isinstance(val, float)
    assert (val > 0) and (val < 1)


def nosym(calc, val):
    """https://www.quantum-espresso.org/Doc/INPUT_PW.html#nosym
    """
    assert isinstance(val, bool)


def lda_plus_u(calc, val):
    """https://www.quantum-espresso.org/Doc/INPUT_PW.html#lda_plus_u
    """
    assert isinstance(val, bool)


def Hubbard_U(calc, val):
    """Take a dictionary of U values associate with each species in
    the SYSTEMS table and assign its value.

    Currently, only element universal U values are supported because of the
    limitations of the ASE write function.

    https://www.quantum-espresso.org/Doc/INPUT_PW.html#Hubbard_U
    """
    assert isinstance(val, dict)

    nspecies = {}
    magmom = calc.atoms.get_initial_magnetic_moments()
    for i, s in enumerate(calc.species):
        smagmom = magmom[np.in1d(calc.symbols, s)]
        nspecies[s] = len(np.unique(smagmom))

    i = 1
    for specie, n in nspecies.items():
        for j in range(n):
            calc.parameters['Hubbard_U({})'.format(i)] = val.get(specie, 0)
            i += 1

    del calc.parameters['Hubbard_U']
    calc.parameters['lda_plus_u'] = True


# PROJWFC variables
def Emin(calc, val):
    """https://www.quantum-espresso.org/Doc/INPUT_PROJWFC.html#emin
    """
    assert isinstance(val, (float, int))

    efermi = calc.get_fermi_level()
    if efermi is not None:
        return val + efermi


def Emax(calc, val):
    """https://www.quantum-espresso.org/Doc/INPUT_PROJWFC.html#emax
    """
    assert isinstance(val, (float, int))

    efermi = calc.get_fermi_level()
    if efermi is not None:
        return val + efermi


def DeltaE(calc, val):
    """https://www.quantum-espresso.org/Doc/INPUT_PROJWFC.html#deltae
    """
    assert isinstance(val, (float, int))


def nguass(calc, val):
    """https://www.quantum-espresso.org/Doc/INPUT_PROJWFC.html#ngauss
    """
    assert isinstance(val, int)
