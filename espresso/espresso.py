from . import siteconfig
from . import validate
import numpy as np
import six
import ase
import warnings
import os


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
                elif isinstance(val, six.string_types):
                    self.params[key] = str(val)
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
            valence = siteconfig.grepy('z valence|z_valence', fname).split()[0]
            nel[species] = int(float(valence))

        nvalence = np.zeros_like(self.symbols, int)
        for i, symbol in enumerate(self.symbols):
            nvalence[i] = nel[symbol]

        # Store the results for nbnd validation.
        self.nvalence, self.nel = nvalence, nel

        return nvalence, nel

    @staticmethod
    def get_fermi_level(outfile='pw.pwo'):
        efermi = siteconfig.grepy('Fermi energy', outfile)
        if efermi is not None:
            efermi = float(efermi.split()[-2])

        return efermi
