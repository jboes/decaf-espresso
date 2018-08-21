from . import siteconfig
import numpy as np
import ase.io as aseio


def read(infile, *args):
    """A wrapper to read the magnetic moment correctly.

    Parameters:
    -----------
    infile : str
        The input file to read from.

    Returns:
    --------
    images : Atoms object | list
        Atoms object or trajectory depending on arguments.
    """
    images = aseio.read(infile, *args)

    if isinstance(images, list):
        with open(infile, 'r') as f:
            lines = f.readlines()

        magmom = []
        for i, line in enumerate(lines):
            if 'absolute magnetization' in line \
               and 'convergence has been achieved' in lines[i + 3]:
                magmom += [line.split('=')[-1].split('Bohr')[0]]

        for i, mag in enumerate(magmom):
            images[i]._calc.results['magmom'] = np.round(float(mag), 2)
    else:
        mag = siteconfig.grepy('absolute magnetization', infile)
        if mag:
            mag = mag.split('=')[-1].split('Bohr')[0]
            images._calc.results['magmom'] = np.round(float(mag), 2)

    return images


def read_input_parameters(infile='pw.pwi'):
    """Return a dictionary of input arguments from an Espresso file.

    Parameters:
    -----------
    infile : str
        Input file to read arguments from.

    Returns:
    --------
    data : dict
        Arguments from an input file.
    """
    data = {}
    with open(infile) as f:
        lines = f.read().split('\n')

        for i, line in enumerate(lines):
            if '=' in line:
                key, value = [_.strip() for _ in line.split('=')]
                value = siteconfig.fortran_to_value(value)
                data[key] = value

            elif 'K_POINTS automatic' in line:
                kpts = np.array(lines[i + 1].split(), dtype=int)
                data['kpts'] = tuple(kpts[:3])
            elif 'K_POINTS gamma' in line:
                data['kpts'] = 1

    return data


def write_projwfc_input(parameters, infile='projwfc.pwi'):
    """Write a projected wave function file.

    Parameters:
    -----------
    parameters : dict
        Input parameters to be written to the file.
    infile : str
        Input file to write to.
    """
    with open(infile, 'w') as f:
        f.write("&PROJWFC\n   {:16} = 'calc'\n   {:16} = '.'\n".format(
            'prefix', 'outdir'))

        for key, value in parameters.items():
            value = siteconfig.value_to_fortran(value)
            f.write('   {:16} = {}\n'.format(key, value))
        f.write('/\n')
