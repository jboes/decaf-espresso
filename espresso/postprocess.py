from .espresso import Espresso
from . import siteconfig
from . import validate
import numpy as np
from ase.io import read
import glob


class PDOS(Espresso):
    """Calculate a projection of wavefunctions for a finished Quantum
    Espresso calculation.

    Assumes that the command is being run in the same directory as a
    completed .pwo file. The atoms object will be read in automatically.
    A calc.tar.gz file must be present for revisited runs.

    Calculation parameters are assumed to be unchanged unless otherwise
    specified in the keyword arguments. If any arguments specific to the
    original files input are changed, such as k-points, these values will
    be automatically updated and an NSCF calculation performed automatically.
    """

    def __init__(self, **kwargs):
        """Read in the atoms object and fermi level from an existing
        calculation and then initialize the Espresso parent class.

        Separates standard input parameters from projection file
        inputs so both can be read from keyword arguments simultaneously.
        Will ignore identical parameters.
        """
        self.atoms = read('pw.pwo')
        self.efermi = self.get_fermi_level()
        super().__init__(self.atoms, **kwargs)

        # Extract the PROJWFC variables
        projwfc_args = {}
        initial_params = self.params.copy()
        for key, value in initial_params.items():
            if key in validate.projwfc_vars:
                projwfc_args[key] = value
                del kwargs[key]

        self.projwfc_args = projwfc_args
        self.params = siteconfig.read_input_parameters()

        # Check for variables that have changed
        self.recalculate = False
        for parameter, value in kwargs.items():
            origional = self.params.get(parameter)

            if origional is not None:
                if isinstance(origional, float):
                    match = np.isclose(origional, value)
                else:
                    match = origional == value

                if not match:
                    self.recalculate = True
                    self.params.update(kwargs)
                    break

    def write_projwfc_input(self, infile='projwfc.pwi'):
        """Write a projected wavefunction file.

        Parameters:
        -----------
        infile : str
            Input file to write to.
        """
        with open(infile, 'w') as f:
            f.write("&PROJWFC\n   {:16} = 'calc'\n   {:16} = '.'\n".format(
                'prefix', 'outdir'))

            for key, value in self.projwfc_args.items():
                value = siteconfig.value_to_fortran(value)
                f.write('   {:16} = {}\n'.format(key, value))
            f.write('/\n')

    def calculate(self):
        """Perform a projwfc.x calculation. If scratch file exists,
        assume an Espresso calculation has been run within the script
        already. Otherwise, looks for calc.tar.gz calculation from a
        previous run in this directory.
        """
        if not self.site.scratch:
            # Starting from completed calculation. Doesn't overwrite save.
            self.site.make_scratch(save_calc=False)
            savefile = self.site.submitdir.joinpath('calc.tar.gz')
            if savefile.exists():
                siteconfig.load_calculator(
                    outdir=self.site.scratch.abspath())
            else:
                raise RuntimeError(
                    'Can not execute a post process calculation '
                    'without a calc.tar.gz file')

        if self.recalculate:
            self.calculate_ncsf()
        self.write_projwfc_input('projwfc.pwi')
        self.site.run('projwfc.x', 'projwfc.pwi', 'projwfc.pwo')

    def calculate_ncsf(self):
        """If the pw.pwi inputs have changed, run a ncsf calculation."""
        self.params['calculation'] = 'nscf'
        self.write_pw_input('nscf.pwi')
        self.site.run(infile='nscf.pwi', outfile='nscf.pwo')
        self.efermi = self.get_fermi_level('nscf.pwo')

    def get_pdos(self):
        """Return the projected densities of states.

        TODO: Most of this is copied from ase-espresso. Not sure what it's
        all of how it works yet.
        """
        channels = {'s': 0, 'p': 1, 'd': 2, 'f': 3}
        output = self.site.scratch.joinpath('calc.pdos_tot')
        states = np.loadtxt(output)
        energies = states[:, 0] - self.efermi

        if len(states[0]) > 3:
            nspin = 2
            dos = [states[:, 1], states[:, 2]]
        else:
            nspin = 1
            dos = states[:, 1]

        # read in projections onto atomic orbitals
        pdos = [{} for _ in self.atoms]
        globstr = '{}/calc.pdos_atm*'.format(self.site.scratch)
        proj = sorted(glob.glob(globstr))

        for infile in proj:
            data = np.genfromtxt(infile)

            split = infile.split('#')
            iatom = int(split[1].split('(')[0]) - 1
            channel = split[2].split('(')[1].rstrip(')').replace('_j', ',j=')
            jpos = channel.find('j=')

            if jpos < 0:
                # ncomponents = 2*l+1+1  (latter for m summed up)
                ncomponents = (2 * channels[channel[0]] + 2) * nspin
            else:
                # ncomponents = 2*j+1+1  (latter for m summed up)
                ncomponents = int(2 * float(channel[jpos + 2:])) + 2

            if channel not in list(pdos[iatom].keys()):
                pdos[iatom][channel] = np.zeros((ncomponents, len(energies)))

                for i in range(ncomponents):
                    pdos[iatom][channel][i] = data[:, (i + 1)]

        return energies, dos, pdos
