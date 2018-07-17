from .espresso import Espresso
from . import siteconfig
from . import validate
import numpy as np
from ase.io import read
import glob


class PDOS(Espresso):
    """Calculate a projection of wavefunctions for a finished Quantum
    Espresso calculation.
    """

    def __init__(self, site=None, **kwargs):
        self.atoms = read('pw.pwo')
        self.efermi = self.get_fermi_level()
        super().__init__(self.atoms, site, **kwargs)

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

    def write_projwfc_input(self, infile):
        with open(infile, 'w') as f:
            f.write("&PROJWFC\n   {:16} = 'calc'\n   {:16} = '.'\n".format(
                'prefix', 'outdir'))

            for key, value in self.projwfc_args.items():
                value = siteconfig.value_to_fortran(value)
                f.write('   {:16} = {}\n'.format(key, value))
            f.write('/\n')

    def calculate(self):
        if not self.site.scratch:
            self.site.make_scratch()
            savefile = self.site.submitdir.joinpath('calc.tar.gz')
            if savefile.exists():
                siteconfig.load_calculator(
                    outdir=self.site.scratch.abspath())
            else:
                raise RuntimeError(
                    'Can not execute a refinement calculation '
                    'without a calc.tar.gz file')

        if self.recalculate:
            self.calculate_ncsf()
        self.write_projwfc_input('projwfc.pwi')
        self.site.run('projwfc.x', 'projwfc.pwi', 'projwfc.pwo')

    def calculate_ncsf(self):
        # We are starting a new instance of the calculation
        self.params['calculation'] = 'nscf'
        self.write_pw_input('nscf.pwi')
        self.site.run(infile='nscf.pwi', outfile='nscf.pwo')

    def get_pdos(self):
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
