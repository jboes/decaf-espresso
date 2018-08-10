from path import Path
import numpy as np
import contextlib
import tempfile
import subprocess
import tarfile
import atexit
import shutil
import re
import os


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
                value = fortran_to_value(value)
                data[key] = value

            elif 'K_POINTS automatic' in line:
                kpts = np.array(lines[i + 1].split(), dtype=int)
                data['kpts'] = tuple(kpts[:3])
            elif 'K_POINTS gamma' in line:
                data['kpts'] = 1

    return data


def grepy(search, filename, instance=-1):
    """Perform a python based grep-like operation for a
    regular expression on a given file.

    Parameters:
    -----------
    search : str
        Regular expression to be found.
    filename : str
        File to be searched within.
    instances : slice
        Index of the arguments to find. If None, return all found results.

    Returns:
    --------
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


@contextlib.contextmanager
def cd(path):
    """Does path management: if the path doesn't exists, create it
    otherwise, move into it until the indentation is broken.

    Parameters
    ----------
    path : str
        Directory path to create and change into.
    """
    cwd = os.getcwd()
    try:
        if not os.path.exists(path):
            os.makedirs(path)
        os.chdir(path)
        yield
    finally:
        os.chdir(cwd)


def load_calculator(infile='calc.tar.gz', outdir='.'):
    """Unpack the contents of calc.save directory which
    was previously compressed.

    Parameters:
    -----------
    infile : str
        Relative input file for compressed save folder.
    outdir : str
        Relative output directory path for calc.save folder.
    """
    with tarfile.open(infile, 'r:gz') as f:
        f.extractall(outdir)


def value_to_fortran(value):
    """Return a Python compatible version of a FORTRAN argument.

    Parameters:
    -----------
    value : bool | int | float | str
        A Python friendly representation of the input value.

    Returns:
    --------
    fortran_value : str
        A FORTRAN argument.
    """
    if isinstance(value, bool):
        fortran_value = '.{}.'.format(str(value).lower())
    elif isinstance(value, float):
        fortran_value = str(value)
        if 'e' not in value:
            fortran_value += 'd0'

    return fortran_value


def fortran_to_value(fortran_value):
    """Return a Python compatible version of a FORTRAN argument.

    Parameters:
    -----------
    fortran_value : str
        A FORTRAN argument.

    Returns:
    --------
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


class SiteConfig():
    """Site configuration holding details about the execution environment
    with methods for retrieving the details from systems variables and
    creating directories.

    Parameters:
    -----------
    scheduler : str
        Name of the scheduler, curretly supports: SLURM, and
        PBS/TORQUE, and LSF
    """

    def __init__(self, scheduler=None, cluster=None):
        self.scheduler = scheduler
        self.cluster = cluster
        self.scratch = None
        self.nnodes = None
        self.nodelist = None
        self.nprocs = None

        if self.scheduler is None:
            self.submitdir = Path(os.getcwd())
            self.jobid = os.getpid()
            return

        lsheduler = self.scheduler.lower()
        if lsheduler == 'slurm':
            self.set_slurm_env()
        elif lsheduler in ['pbs', 'torque']:
            self.set_pbs_env()
        elif lsheduler == 'lbs':
            self.set_lbs_env()

    @classmethod
    def check_scheduler(cls):
        """Check for appropriate environment variables for the name
        of the cluster being used. Returns None for both if no
        supported schedule is found.

        Returns:
        --------
        cls : SiteConfig object
            Runs SiteConfig with the found arguments.
        """
        check_shedulers = {
            'SLURM': 'SLURM_CLUSTER_NAME',
            'PBS': 'PBS_SERVER',
            'LBS': 'LSB_EXEC_CLUSTER'}

        scheduler = None
        for sched, ev in check_shedulers.items():
            cluster = os.getenv(ev)
            if cluster:
                scheduler = sched
                break

        return cls(scheduler, cluster)

    def set_slurm_env(self):
        """Set the attributes necessary to run the job based on the
        enviromental variables associated with SLURM scheduler.
        """
        self.jobid = os.getenv('SLURM_JOB_ID')
        self.submitdir = Path(os.getenv('SLURM_SUBMIT_DIR'))

        tpn = int(os.getenv('SLURM_TASKS_PER_NODE').split('(')[0])

        cmd = ['scontrol', 'show', 'hostnames',
               os.getenv('SLURM_JOB_NODELIST')]
        self.nodelist = subprocess.check_output(cmd).decode().split('\n')[:-1]
        proclist = sorted(self.nodelist * tpn)

        self.nnodes = int(os.getenv('SLURM_JOB_NUM_NODES'))
        self.nprocs = len(proclist)

    def set_lbs_env(self):
        """Set the attributes necessary to run the job based on the
        enviromental variables associated with LBS scheduler.
        """
        self.jobid = os.getenv('LSB_BATCH_JID')
        self.submitdir = Path(os.getenv('LS_SUBCWD'))

        nodefile = os.getenv('LSB_HOSTS').split()
        procs = [_ for _ in nodefile]

        self.nodelist = list(set(procs))
        self.nnodes = len(self.nodelist)
        self.nprocs = len(procs)

    def set_pbs_env(self):
        """Set the attributes necessary to run the job based on the
        enviromental variables associated with PBS/TORQUE scheduler.
        """
        self.jobid = os.getenv('PBS_JOBID')
        self.submitdir = Path(os.getenv('PBS_O_WORKDIR'))

        nodefile = os.getenv('PBS_NODEFILE')
        with open(nodefile, 'r') as f:
            procs = [_.strip() for _ in f.readlines()]

        self.nodelist = list(set(procs))
        self.nnodes = len(self.nodelist)
        self.nprocs = len(procs)

    def make_scratch(self, save_calc=True):
        """Create a scratch directory on each calculation node if batch mode
        or a single scratch directory otherwise. Will attempt to call
        from $L_SCRATCH_JOB variable, then /tmp, then use the
        submission directory.

        This function will automatically cleanup upon exiting Python.

        Parameters:
        -----------
        save_calc : bool
            Whether to save the calculation folder or not.
        """
        scratch_paths = [os.getenv('ESPRESSO_SCRATCH', '/tmp'), self.submitdir]
        for scratch in scratch_paths:
            if os.path.exists(scratch):
                node_scratch = Path(scratch)
                break

        scratch = Path(tempfile.mkdtemp(
            prefix='qe_{}'.format(self.jobid), suffix='_scratch',
            dir=node_scratch)).abspath()

        with cd(self.submitdir):
            if self.scheduler:
                cmd = self.get_exe_command('mkdir -p {}'.format(scratch))
                subprocess.call(cmd)
            else:
                scratch.makedirs_p()

        self.scratch = scratch
        atexit.register(self.clean, save_calc)

    def get_exe_command(self, program, workdir=None):
        """Return a command as list to execute subprocess through
        a supplied argument. If a workdir is provided, assume
        execution per processor, otherwise, per host.

        Parameters:
        -----------
        program : str
            The full command line program to be executed using subprocess
        workdir : str
            A path to act as the working directory.

        Returns:
        --------
        command : list or str (N,)
            The list of arguments to be passed to subprocess.
        """
        exe, host, nproc, wd = ['mpiexec'], '-host', '-np', '-wdir'

        # SPECIAL SERVER EXECUTABLE CASES ARE HANDLED HERE
        if self.cluster == 'edison':
            exe, host, nproc, wd = ['srun'], '-w', '-n', '-D'
        elif self.cluster == 'slac' and workdir:
            exe = ['pam', '-g', '/afs/slac/g/suncat/bin/suncat-tsmpirun',
                   '-x', 'LD_LIBRARY_PATH']
        elif self.cluster == 'slac':
            exe = ['mpiexec', '--mca', 'orte_rsh_agent',
                   '/afs/slac.stanford.edu/package/lsf/bin.slac/gmmpirun_lsgrun.sh']
            init = exe + ['ls', self.submitdir]
            subprocess.call(init)

        # This indicates per-processor MPI run
        if workdir:
            command = exe + [wd, workdir]
        # Otherwise, per-host MPI run
        else:
            command = exe + [nproc, str(self.nnodes),
                             host, ','.join(self.nodelist)]
        command += program.split()

        return command

    def run(self, exe='pw.x', infile='pw.pwi', outfile='pw.pwo'):
        """Run an Espresso executable with subprocess. The executable will
        attempt to automatically assign an intelligent npool value
        for efficient parallelization.

        Parameters:
        -----------
        exe : str
            The Espresso executable command to be run.
        infile : str
            Name of the input file to be used for the executable.
        outfile : str
            Name of the output file to be used for the executable.

        Returns:
        --------
        state : int
            The output state of the subprocess executed command.
        """
        mypath = os.path.abspath(os.path.dirname(__file__))
        exedir = subprocess.check_output(['which', exe]).decode()

        # Copy the input file to the scratch directory.
        inp = self.submitdir.joinpath(infile)
        inp.copy(self.scratch)

        if self.scheduler:
            # Automatically assign npool for parallelization
            parflags = ''
            kpts = read_input_parameters()['kpts']
            if self.nprocs > 1 and kpts > self.nprocs:
                parflags += '-npool {}'.format(self.nprocs)

            command = self.get_exe_command(
                '{} {} -in {}'.format(exe, parflags, infile), self.scratch)
        else:
            command = [exe, '-in', infile]

        title = '{:<14}: {}\n{:<14}: {}{:<14}: {}\n'.format(
            'python dir', mypath,
            'espresso dir', exedir,
            'executable', ' '.join(command))

        # This will remove the old output file by default.
        output = self.submitdir.joinpath(outfile)
        with open(output, 'w') as f:
            f.write(title)

        with cd(self.scratch):
            with open(output, 'ab') as f:
                state = subprocess.call(command, stdout=f)

        if state != 0:
            if grepy('JOB DONE.', outfile):
                pass
            elif grepy('is really the minimum energy structure', outfile):
                pass
            else:
                # Read the error message
                error_message = []
                with open(outfile, 'r') as f:
                    for line in f:
                        # Capture the message between %%%
                        if '%%%%%%%%%%%%%%' in line:
                            error_message += [line]
                            if len(error_message) > 1:
                                break
                        elif error_message:
                            error_message += [line]

                raise RuntimeError('pw.x returned a nonzero exit state:\n'
                                   '{}'.format(''.join(error_message)))

        return state

    def clean(self, save_calc=True):
        """Remove temporary files and directories."""
        os.chdir(self.submitdir)

        calc = self.scratch.joinpath('calc.save')
        save = self.submitdir.joinpath('calc.tar.gz')

        if os.path.exists(calc) and save_calc:
            with tarfile.open(save, 'w:gz') as f:
                f.add(calc, arcname=calc.basename())

        shutil.rmtree(self.scratch)
