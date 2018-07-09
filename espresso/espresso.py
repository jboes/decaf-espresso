from . import siteconfig
from . import validate
import ase
import atexit
import subprocess
import numpy as np
import tarfile
import warnings
import shutil
import os
defaults = validate.variables


class Espresso(ase.calculators.calculator.FileIOCalculator):
    """Decaf Espresso

    A Light weight ASE interface for Quantum Espresso
    """

    implemented_properties = [
        'energy', 'forces', 'stress', 'magmom', 'magmoms']

    def __init__(
            self,
            atoms,
            **kwargs):
        atoms.set_calculator(self)
        self.params = kwargs.copy()
        self.results = {}

        self.site = siteconfig.SiteConfig.check_scheduler()
        self.natoms = len(self.atoms)
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
                if key not in defaults:
                    warnings.warn('Not a valid key {}'.format(key))
                    del self.params[key]
                else:
                    warnings.warn('No validation for {}'.format(key))

    def initialize(self):
        """Create the input file to start the calculation."""
        if self.get_param('dipfield') is not None:
            self.params['tefield'] = True

        if self.get_param('tstress') is not None:
            self.params['tprnfor'] = True

        self.params['nat'] = len(self.symbols)
        self.params['ntyp'] = len(self.species)

        # Apply any remaining default settings
        for key, value in defaults.items():
            setting = self.get_param(key)
            if setting is not None:
                self.params[key] = setting

        # ASE format for the pseudopotential file location.
        self.params['pseudopotentials'] = {}
        for species in self.species:
            self.params['pseudopotentials'][species] = '{}.UPF'.format(species)

        ase.io.write('pwcsf.pwi', self.atoms, **self.params)

    def calculate(self, atoms, properties=['energy'], changes=None):
        """Perform a calculation."""
        self.initialize()
        self.create_outdir()
        self.run()

        atoms = ase.io.read('pwcsf.pwo')
        self.set_results(atoms)

    def set_atoms(self, atoms):
        self.atoms = atoms.copy()

    def set_results(self, atoms):
        self.results = atoms._calc.results

    def get_param(self, parameter):
        """Return the parameter associated with a calculator,
        otherwise, return the default value.
        """
        value = self.params.get(parameter, defaults[parameter])

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

        # Store the results for later.
        self.nvalence, self.nel = nvalence, nel

        return nvalence, nel

    def get_fermi_level(self):
        efermi = siteconfig.grepy('pwcsf.pwo', 'Fermi energy')
        if efermi is not None:
            efermi = float(efermi.split()[-2])

        return efermi

    def load_calculator(self, filename='calc.tar.gz'):
        """Unpack the contents of calc.save directory."""
        with tarfile.open(filename, 'r:gz') as f:
            f.extractall()

    def create_outdir(self):
        """Create the necessary directory structure to run the calculation and
        assign file names
        """
        self.localtmp = self.site.make_localtmp('')
        self.scratch = self.site.make_scratch()
        self.log = self.localtmp.joinpath('pwcsf.pwo')

        atexit.register(self.clean)

    @siteconfig.preserve_cwd
    def run(self):
        """Execute the expresso program `pw.x`"""
        mypath = os.path.abspath(os.path.dirname(__file__))
        exedir = subprocess.check_output(['which', 'pw.x']).decode()
        psppath = self.get_param('pseudo_dir')

        title = '{:<14}: {}\n{:<14}: {}{:<14}: {}\n'.format(
            'python dir', mypath, 'espresso dir',
            exedir, 'psuedo dir', psppath)

        # This will remove the old log file.
        with open(self.log, 'w') as f:
            f.write(title)

        if self.site.batchmode:
            self.localtmp.chdir()
            shutil.copy(self.localtmp.joinpath('pwcsf.pwi'), self.scratch)

            command = self.site.get_proc_mpi_command(
                self.scratch, 'pw.x ' + self.parflags + ' -in pwcsf.pwi')

            with open(self.log, 'ab') as f:
                exitcode = subprocess.call(command, stdout=f)
            if exitcode != 0:
                raise RuntimeError('something went wrong:', exitcode)

        else:
            pwinp = self.localtmp.joinpath('pwcsf.pwi')
            shutil.copy(pwinp, self.scratch)
            command = ['pw.x', '-in', 'pwcsf.pwi']

            self.scratch.chdir()
            with open(self.log, 'ab') as f:
                exitcode = subprocess.call(command, stdout=f)

    def clean(self):
        """Remove the temporary files and directories"""
        os.chdir(self.site.submitdir)

        savefile = self.scratch.joinpath('calc.save')
        newsave = self.site.submitdir.joinpath('calc.tar.gz')

        if os.path.exists(newsave):
            newsave.remove()

        if os.path.exists(savefile):
            # Remove wavecars by default
            for f in savefile.files('wfc*.dat'):
                f.remove()

            with tarfile.open(newsave, 'w:gz') as f:
                f.add(savefile, arcname=savefile.basename())

        self.scratch.rmtree_p()

        if (hasattr(self.site, 'mpdshutdown')
           and 'QEASE_MPD_ISSHUTDOWN' not in list(os.environ.keys())):
            os.environ['QEASE_MPD_ISSHUTDOWN'] = 'yes'
            os.system(self.site.mpdshutdown)
