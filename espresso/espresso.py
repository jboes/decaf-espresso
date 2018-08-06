from . import siteconfig
from . import validate
import numpy as np
import six
import ase
import warnings
import os


class Espresso(ase.calculators.calculator.FileIOCalculator):
    """Decaf Espresso - A Light weight ASE interface for Quantum Espresso.

    Validation of arguments present in validate.py will be performed
    automatically.

    Configuration of site executables and scratch directories will be
    performed automatically with siteconfig.py
    """

    implemented_properties = [
        'energy', 'forces', 'stress', 'magmom', 'magmoms']

    def __init__(
            self,
            atoms=None,
            **kwargs):
        """Initialize the calculators by validating the input key word
        arguments corresponding to valid Quantum Espresso arguments:
        https://www.quantum-espresso.org/Doc/INPUT_PW.html

        Geometric aspects of the calculation are defined by the atoms
        object, so geometry related arguments are ignored.

        Parameters:
        -----------
        atoms : Atoms object
            ASE atoms to attach the calculator with.
        """
        self.params = kwargs.copy()
        self.defaults = validate.variables
        self.results = {}
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
        """Create the input file to start the calculation. Defines unspecified
        defaults as defined in validate.py.

        Parameters:
        -----------
        infile : str
            Name of the input file which will be written.
        """
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
        self.write_pw_input('pw.pwi')

        self.site.make_scratch()
        self.site.run(infile='pw.pwi', outfile='pw.pwo')

        atoms = ase.io.read('pw.pwo')
        self.set_atoms(atoms)
        self.set_results(atoms)

    def set_results(self, atoms):
        """Set the results of the calculator."""
        self.results = atoms._calc.results

    def set_atoms(self, atoms):
        """Set the atoms object to the calculator."""
        self.atoms = atoms.copy()

    def get_param(self, parameter):
        """Return the parameter associated with a calculator,
        otherwise, return the default value.

        Parameters:
        -----------
        parameter : str
            Name of the parameter to retrieve the value of.

        Returns:
        --------
        value : bool | int | float | str | None
            The parameter value specified by the user, or None
        """
        value = self.params.get(parameter, self.defaults[parameter])

        return value

    def get_nvalence(self):
        """Get number of valence electrons from pseudopotential file associated
        with an atoms object.

        Returns:
        --------
        nvalence : ndarray (N,)
            Number of valence electrons associated with each atom N.
        nel : dict
            Number of electrons associated with species in the atoms object.
        """
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
        """Return the fermi level in eV from a completed calculation file.

        Parameters:
        -----------
        outfile : str
            The completed calculation file to read the fermi level from.

        Returns:
        --------
        efermi : float
            The fermi energy in eV.
        """
        efermi = siteconfig.grepy('Fermi energy', outfile)
        if efermi is not None:
            efermi = float(efermi.split()[-2])

        return efermi
