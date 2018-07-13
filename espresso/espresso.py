from . import siteconfig
from . import validate
import numpy as np
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
        ignored_keys = ['prefix', 'outdir', 'restart_mode', 'ibrav', 'celldm',
                        'A', 'B', 'C', 'cosAB', 'cosAC', 'cosBC', 'nat',
                        'ntyp']

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

    def initialize(self):
        """Create the input file to start the calculation."""
        for key, value in self.defaults.items():
            setting = self.get_param(key)
            if setting is not None:
                self.params[key] = setting

        # ASE format for the pseudopotential file location.
        self.params['pseudopotentials'] = {}
        for species in self.species:
            self.params['pseudopotentials'][species] = '{}.UPF'.format(species)

        ase.io.write(self.infile, self.atoms, **self.params)

    def calculate(self, atoms, properties=['energy'], changes=None):
        """Perform a calculation."""
        self.initialize()

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

    def get_fermi_level(self):
        efermi = siteconfig.grepy(self.outfile, 'Fermi energy')
        if efermi is not None:
            efermi = float(efermi.split()[-2])

        return efermi


class QEProjection(Espresso):
    """Calculate a projection of wavefunctions for a finished Quantum
    Espresso calculation.

    THIS IS STILL A WORK IN PROGRESS
    """

    def __init__(self, **kwargs):
        super().__init__(None, None, **kwargs)
        self.dos_energies = None
        self.dos_total = None
        self.pdos = None

        if self.get_param('calculation') in ['ncsf', 'bands']:
            self.infile = 'ncsf.in'
            self.initialize()

            self.site.make_scratch()
            self.site.run(infile=self.infile, outfile=self.outfile)

            atoms = ase.io.read(self.outfile)
            self.set_atoms(atoms)
            self.set_results(atoms)

    def write_input(self):
        with open(self.infile, 'w') as f:
            f.write("&PROJWFC\n   {:16} = 'calc'\n   {:16} = '.'\n".format(
                'prefix', 'outdir'))

            for key, value in self.params.items():
                value = siteconfig.fortran_conversion(value)
                f.write('   {:16} = {}\n'.format(key, value))
            f.write('/\n')

    def run(self):
        """Check if this is a refinement calculation
        If so, we need the calculation file
        """

        # COMPLETELY NON-FUNCTIONAL
        calc = siteconfig.grepy(pwinp, 'calculation')
        if calc is not None:
            calc = calc.split("'")[-2]

        if calc in ['nscf', 'bands']:
            savefile = self.sitesubmitdir.joinpath('calc.tar.gz')
            if savefile.exists():
                siteconfig.load_calculator(
                    outdir=self.site.scratch.abspath())
            else:
                raise RuntimeError('Can not executing a refinement '
                                   'calculation without a calc.tar.gz file')

    def calculate(self):
        self.initialize()
        self.site.run('projwfc.x', self.infile, self.outfile)

    def read_pdos(self):
        output = self.site.scratch.joinpath('calc.pdos_tot')
        dos = np.loadtxt(output)
        self.dos_energies = dos[:, 0] - self.efermi

        if len(dos[0]) > 3:
            nspin = 2
            self.dos_total = [dos[:, 1], dos[:, 2]]
        else:
            nspin = 1
            self.dos_total = dos[:, 1]

        npoints = len(self.dos_energies)
        channels = {'s': 0, 'p': 1, 'd': 2, 'f': 3}

        # read in projections onto atomic orbitals
        self.pdos = [{} for i in range(self.natoms)]
        globstr = '{}/calc.pdos_atm*'.format(self.site.scratch)
        proj = glob.glob(globstr).sort()
        for i, inpfile in enumerate(proj):
            pdosinp = np.genfromtxt(inpfile)
            spl = inpfile.split('#')
            iatom = int(spl[1].split('(')[0]) - 1
            channel = spl[2].split('(')[1].rstrip(')').replace('_j', ',j=')
            jpos = channel.find('j=')

            if jpos < 0:
                # ncomponents = 2*l+1 +1  (latter for m summed up)
                ncomponents = (2*channels[channel[0]]+2) * nspin
            else:
                # ncomponents = 2*j+1 +1  (latter for m summed up)
                ncomponents = int(2.*float(channel[jpos+2:])) + 2

            if channel not in list(self.pdos[iatom].keys()):
                self.pdos[iatom][channel] = np.zeros(
                    (ncomponents, npoints), float)

                for j in range(ncomponents):
                    self.pdos[iatom][channel][j] += pdosinp[:, (j+1)]

        return self.dos_energies, self.dos_total, self.pdos


