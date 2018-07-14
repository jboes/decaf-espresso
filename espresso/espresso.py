from . import siteconfig
from . import validate
import numpy as np
from ase.io import read
import ase
import warnings
import os
import glob


class Espresso(ase.calculators.calculator.FileIOCalculator):
    """Decaf Espresso

    A Light weight ASE interface for Quantum Espresso
    """

    implemented_properties = [
        'energy', 'forces', 'stress', 'magmom', 'magmoms']

    def __init__(
            self,
            atoms=None,
            site=None,
            **kwargs):
        self.params = kwargs.copy()
        self.defaults = validate.variables
        self.results = {}
        self.infile = 'pw.pwi'
        self.outfile = 'pw.pwo'

        self.site = site
        if site is None:
            self.site = siteconfig.SiteConfig.check_scheduler()

        if atoms:
            atoms.set_calculator(self)
            self.symbols = self.atoms.get_chemical_symbols()
            self.species = np.unique(self.symbols)

        # Certain keys are used for fixed IO features or atoms object
        # information. For calculation consistency, user input is ignored.
        ignored_keys = ['prefix', 'outdir', 'ibrav', 'celldm', 'A', 'B', 'C',
                        'cosAB', 'cosAC', 'cosBC', 'nat', 'ntyp']

        # Run validation checks
        init_params = self.params.copy()
        for key, val in init_params.items():

            if key in ignored_keys:
                del self.params[key]
            elif key in validate.__dict__:
                f = validate.__dict__[key]
                new_val = f(self, val)

                # Used to convert values from eV to Ry
                if new_val is not None:
                    self.params[key] = new_val
            else:
                warnings.warn('No validation for {}'.format(key))

    def write_pw_input(self, infile='pw.pwi'):
        """Create the input file to start the calculation."""
        for key, value in self.defaults.items():
            setting = self.get_param(key)
            if setting is not None:
                self.params[key] = setting

        # ASE format for the pseudopotential file location.
        self.params['pseudopotentials'] = {}
        for species in self.species:
            self.params['pseudopotentials'][species] = '{}.UPF'.format(species)

        ase.io.write(infile, self.atoms, **self.params)

    def calculate(self, atoms, properties=['energy'], changes=None):
        """Perform a calculation."""
        self.write_pw_input(self.infile)

        self.site.make_scratch()
        self.site.run(infile=self.infile, outfile=self.outfile)

        atoms = ase.io.read(self.outfile)
        self.set_atoms(atoms)
        self.set_results(atoms)

    def set_results(self, atoms):
        self.results = atoms._calc.results

    def set_atoms(self, atoms):
        self.atoms = atoms.copy()

    def get_param(self, parameter):
        """Return the parameter associated with a calculator,
        otherwise, return the default value.
        """
        value = self.params.get(parameter, self.defaults[parameter])

        return value

    def get_nvalence(self):
        """Get number of valence electrons from pseudopotential files"""
        if hasattr(self, 'nel'):
            return self.nvalence, self.nel

        nel = {}
        for species in self.species:
            fname = os.path.join(
                self.get_param('pseudo_dir'), '{}.UPF'.format(species))
            valence = siteconfig.grepy(fname, 'z valence|z_valence').split()[0]
            nel[species] = int(float(valence))

        nvalence = np.zeros_like(self.symbols, int)
        for i, symbol in enumerate(self.symbols):
            nvalence[i] = nel[symbol]

        # Store the results for nbnd validation.
        self.nvalence, self.nel = nvalence, nel

        return nvalence, nel

    @staticmethod
    def get_fermi_level(outfile='pw.pwo'):
        efermi = siteconfig.grepy(outfile, 'Fermi energy')
        if efermi is not None:
            efermi = float(efermi.split()[-2])

        return efermi


class QEpdos(Espresso):
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
        self.params = read_input_parameters()

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


def read_input_parameters():
    data = {}
    with open('pw.pwi') as f:
        lines = f.read().split('\n')

        for i, line in enumerate(lines):
            if '=' in line:
                key, value = [_.strip() for _ in line.split('=')]
                value = siteconfig.fortran_to_value(value)
                data[key] = value

            elif 'K_POINTS automatic' in line:
                kpts = np.array(lines[i + 1].split(), dtype=int)
                data['kpts'] = tuple(kpts[:3])

    return data
