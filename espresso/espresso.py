from . import siteconfig
from . import validate
from . import io
import numpy as np
import warnings
import glob
import six
import ase
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
            site=None,
            **kwargs):
        """Initialize the calculators by validating the input key word
        arguments corresponding to valid Quantum Espresso arguments:
        https://www.quantum-espresso.org/Doc/INPUT_PW.html

        Geometric aspects of the calculation are defined by the atoms
        object, so geometry related arguments are ignored.

        Parameters
        ----------
        atoms : Atoms object
            ASE atoms to attach the calculator with.
        """
        self.name = 'decaf-espresso'
        self.parameters = kwargs.copy()
        self.defaults = validate.variables
        self.results = {}
        self.site = site
        if self.site is None:
            self.site = siteconfig.SiteConfig.check_scheduler()

        if atoms is not None:
            atoms.set_calculator(self)
            self.symbols = self.atoms.get_chemical_symbols()
            self.species = np.unique(self.symbols)

        # Certain keys are used for fixed IO features or atoms object
        # information. For calculation consistency, user input is ignored.
        ignored_keys = ['prefix', 'outdir', 'ibrav', 'celldm', 'A', 'B', 'C',
                        'cosAB', 'cosAC', 'cosBC', 'nat', 'ntyp']

        # Run validation checks
        init_params = self.parameters.copy()
        for key, val in init_params.items():

            if key in ignored_keys:
                del self.parameters[key]
            elif key in validate.__dict__:
                f = validate.__dict__[key]
                new_val = f(self, val)

                # Used to convert values from eV to Ry
                if new_val is not None:
                    self.parameters[key] = new_val
                elif isinstance(val, six.string_types):
                    self.parameters[key] = str(val)
            else:
                warnings.warn('No validation for {}'.format(key))

    def write_input(self, infile='pw.pwi'):
        """Create the input file to start the calculation. Defines unspecified
        defaults as defined in validate.py.

        Note: This overwrites the existing ASE calculator function.

        Parameters
        ----------
        infile : str
            Name of the input file which will be written.
        """
        for key, value in self.defaults.items():
            setting = self.get_param(key)
            if setting is not None:
                self.parameters[key] = setting

        # ASE format for the pseudopotential file location.
        self.parameters['pseudopotentials'] = {}
        for species in self.species:
            self.parameters['pseudopotentials'][species] = '{}.UPF'.format(
                species)

        ase.io.write(infile, self.atoms, **self.parameters)

    def calculate(self, atoms, properties=['energy'], changes=None):
        """Perform a calculation."""
        self.write_input('pw.pwi')

        self.site.make_scratch()
        self.site.run(infile='pw.pwi', outfile='pw.pwo')

        relaxed_atoms = io.read('pw.pwo')

        atoms.arrays = relaxed_atoms.arrays
        self.set_results(relaxed_atoms)

    def set_results(self, atoms):
        """Set the results of the calculator."""
        self.results = atoms._calc.results

    def set_atoms(self, atoms):
        """Set the atoms object to the calculator."""
        self.atoms = atoms

    def get_param(self, parameter):
        """Return the parameter associated with a calculator,
        otherwise, return the default value.

        Parameters
        ----------
        parameter : str
            Name of the parameter to retrieve the value of.

        Returns
        -------
        value : bool | int | float | str | None
            The parameter value specified by the user, or None
        """
        value = self.parameters.get(parameter, self.defaults.get(parameter))

        return value

    def get_nvalence(self):
        """Get number of valence electrons from pseudopotential file associated
        with an atoms object.

        Returns
        -------
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
            valence = io.grepy('z valence|z_valence', fname).split()[0]
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

        Parameters
        ----------
        outfile : str
            The completed calculation file to read the fermi level from.

        Returns
        -------
        efermi : float
            The fermi energy in eV.
        """
        efermi = io.grepy('Fermi energy', outfile)
        if efermi is not None:
            efermi = float(efermi.split()[-2])

        return efermi

    def get_pdos(self, parameters=None, update_projections=False):
        """Wrapper for the PDOS class for getting the projected density of
        states.

        Parameters
        ----------
        parameters : dict
            Parameters to pass to PDOS class for a projwfc calculation.
        update_projections : bool
            Whether to attach an updated estimate of spin derived parameters
            to the atoms object (magmoms and charge).

        Returns
        -------
        pdos : list (3,)
            Energy range, total DOS, and projected DOS of a calculation.
        """
        if parameters is None:
            parameters = {}
        proj = PDOS(self.site, **parameters)
        pdos = proj.get_pdos()

        if update_projections:
            magmoms, charge = proj.get_updated_projections()
            for k, v in {'magmoms': magmoms, 'charge': charge}.items():
                if k in self.results:
                    self.results[k] = v

        return pdos


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

    def __init__(self, site=None, **kwargs):
        """Read in the atoms object and fermi level from an existing
        calculation and then initialize the Espresso parent class.

        Separates standard input parameters from projection file
        inputs so both can be read from keyword arguments simultaneously.
        Will ignore identical parameters.
        """
        super().__init__(None, site, **kwargs)
        self.efermi = self.get_fermi_level()
        self.atoms = io.read('pw.pwo')
        self.symbols = self.atoms.get_chemical_symbols()
        self.species = np.unique(self.symbols)

        # Extract the PROJWFC variables
        projwfc_args = {}
        initial_params = self.parameters.copy()
        for key, value in initial_params.items():
            if key in validate.projwfc_vars:
                projwfc_args[key] = value
                del kwargs[key]

        self.projwfc_args = projwfc_args
        self.parameters = io.read_input_parameters()

        # Check for variables that have changed
        self.recalculate = False
        for parameter, value in kwargs.items():
            origional = self.parameters.get(parameter)

            if origional is not None:
                if isinstance(origional, float):
                    match = np.isclose(origional, value)
                else:
                    match = origional == value

                if not match:
                    self.recalculate = True
                    self.parameters.update(kwargs)
                    break

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
        io.write_projwfc_input(self.projwfc_args, 'projwfc.pwi')
        self.site.run('projwfc.x', 'projwfc.pwi', 'projwfc.pwo')

    def calculate_ncsf(self):
        """If the pw.pwi inputs have changed, run a ncsf calculation."""
        self.parameters['calculation'] = 'nscf'
        self.write_input('nscf.pwi')
        self.site.run(infile='nscf.pwi', outfile='nscf.pwo')
        self.efermi = self.get_fermi_level('nscf.pwo')

    def get_pdos(self):
        """Return the projected densities of states.

        TODO: Most of this is copied from ase-espresso. Not sure what it's
        all of how it works yet.
        """
        channels = {'s': 0, 'p': 1, 'd': 2, 'f': 3}
        if not self.site.scratch:
            self.calculate()
        output = self.site.scratch.joinpath('calc.pdos_tot')
        if not os.path.exists(output):
            self.calculate()

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

        self.pdos = [energies, dos, pdos]
        return energies, dos, pdos

    def get_updated_projections(self):
        """Get a refined estimate of the magnetic moments and charge
        from the projected density of states.

        Returns
        -------
        magmoms : ndarray (N,)
            Projected magnetic moment of individual atoms.
        charge : ndarray (N,)
            Projected charge of individual atoms.
        """
        pdos = self.pdos
        if pdos is None:
            pdos = self.get_pdos()

        fs = pdos[0] <= 0
        energies = pdos[0][fs]

        spins = np.zeros((2, len(pdos[2])))
        for i in range(spins.shape[1]):
            for band in pdos[2][i]:
                spins[0][i] += np.trapz(pdos[2][i][band][0][fs], x=energies)
                spins[1][i] += np.trapz(pdos[2][i][band][1][fs], x=energies)
        magmoms = np.round(-np.diff(spins, axis=0)[0], 2)
        charge = self.get_nvalence()[0] - np.round(spins.sum(axis=0), 2)

        return magmoms, charge
