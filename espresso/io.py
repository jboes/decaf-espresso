import numpy as np
import ase.io as aseio
import re
import os


def read(infile, *args):
    """A wrapper to read the magnetic moment correctly.

    Parameters
    ----------
    infile : str
        The input file to read from.

    Returns
    -------
    images : Atoms object | list
        Atoms object or trajectory depending on arguments.
    """
    images = aseio.read(infile, *args)

    read_ldu_nmag = grepy('atom [\d ]{4}   Tr\[ns\(na\)\]', infile, ':')
    if read_ldu_nmag:
        magmom_ind = np.unique([int(_.split()[1]) for _ in read_ldu_nmag]) - 1
        n = len(magmom_ind)

        read_ldu_mag = grepy('atomic mag.', infile, ':')
        magmoms = np.round([float(_.split()[-1]) for _ in read_ldu_mag], 4)

    if isinstance(images, list):
        with open(infile, 'r') as f:
            lines = f.readlines()

        magmom = []
        for i, line in enumerate(lines):
            if 'total magnetization' in line \
               and 'convergence has been achieved' in lines[i + 3]:
                magmom += [line.split()[-3]]

        for i, atoms in enumerate(images):
            atoms.calc.results['magmom'] = np.round(float(magmom[i]), 2)

            if read_ldu_nmag:
                mag = atoms.calc.results['magmoms']
                mag[magmom_ind] = magmoms[(i+1)*n + n:(i+1)*n + n*2]

        if read_ldu_nmag:
            images[-1].calc.results['magmoms'][magmom_ind] = magmoms[-n:]
    else:
        magmom = grepy('absolute magnetization', infile)
        if magmom:
            magmom = magmom.split('=')[-1].split('Bohr')[0]
            images.calc.results['magmom'] = np.round(float(magmom), 2)

        if read_ldu_nmag:
            images.calc.results['magmoms'][magmom_ind] = magmoms[-n:]

    return images


def read_input_parameters(infile='pw.pwi'):
    """Return a dictionary of input arguments from an Espresso file.

    Parameters
    ----------
    infile : str
        Input file to read arguments from.

    Returns
    -------
    data : dict
        Arguments from an input file.
    """
    data = {}
    with open(infile) as f:
        lines = f.read().split('\n')

        for i, line in enumerate(lines):
            if '=' in line:
                key, value = [_.strip() for _ in line.split('=')]
                value = fortran_to_value(value)
                data[key] = value

            elif 'K_POINTS automatic' in line:
                kpts = np.array(lines[i + 1].split(), dtype=int)
                data['kpts'] = tuple(kpts[:3])
            elif 'K_POINTS gamma' in line:
                data['kpts'] = 1

    return data


def write_projwfc_input(parameters, infile='projwfc.pwi'):
    """Write a projected wave function file.

    Parameters
    ----------
    parameters : dict
        Input parameters to be written to the file.
    infile : str
        Input file to write to.
    """
    with open(infile, 'w') as f:
        f.write("&PROJWFC\n   {:16} = 'calc'\n   {:16} = '.'\n".format(
            'prefix', 'outdir'))

        for key, value in parameters.items():
            value = value_to_fortran(value)
            f.write('   {:16} = {}\n'.format(key, value))
        f.write('/\n')


def grepy(search, filename, instance=-1):
    """Perform a python based grep-like operation for a
    regular expression on a given file.

    Parameters
    ----------
    search : str
        Regular expression to be found.
    filename : str
        File to be searched within.
    instances : slice
        Index of the arguments to find. If None, return all found results.

    Returns
    -------
    results : list of str (N,) | None
        All requested instances of a given argument.
    """
    if not os.path.exists(filename):
        return None

    results = []
    with open(filename, 'r') as f:
        for line in f:
            if re.search(search, line, re.IGNORECASE):
                results += [line]

    if not results:
        return None

    if isinstance(instance, int):
        return results[instance]
    else:
        return results


def value_to_fortran(value):
    """Return a Python compatible version of a FORTRAN argument.

    Parameters
    ----------
    value : bool | int | float | str
        A Python friendly representation of the input value.

    Returns
    -------
    fortran_value : str
        A FORTRAN argument.
    """
    if isinstance(value, bool):
        fortran_value = '.{}.'.format(str(value).lower())
    elif isinstance(value, float):
        fortran_value = str(value)
        if 'e' not in fortran_value:
            fortran_value += 'd0'

    return fortran_value


def fortran_to_value(fortran_value):
    """Return a Python compatible version of a FORTRAN argument.

    Parameters
    ----------
    fortran_value : str
        A FORTRAN argument.

    Returns
    -------
    value : bool | int | float | str
        A Python friendly representation of the input value.
    """
    if fortran_value == '.true.':
        return True
    elif fortran_value == '.false.':
        return False

    try:
        value = int(fortran_value)
    except(ValueError):
        try:
            value = float(fortran_value)
        except(ValueError):
            value = fortran_value.strip("'")

    return value
