from six import with_metaclass
from path import Path
import contextlib
import itertools
import os
import tempfile
import subprocess
import re
import hostlist
import tarfile
import atexit


def grepy(filename, search, instance=-1):
    if not os.path.exists(filename):
        return None

    results = []
    with open(filename, 'r') as f:
        for line in f:
            if re.search(search, line, re.IGNORECASE):
                results += [line]

    if not results:
        return None

    if instance:
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
    """Unpack the contents of calc.save directory."""
    with tarfile.open(infile, 'r:gz') as f:
        f.extractall(outdir)


def value_to_fortran(value):
    if isinstance(value, bool):
        value = '.{}.'.format(str(value).lower())
    elif isinstance(value, float):
        value = str(value)
        if 'e' not in value:
            value += 'd0'

    return value


def fortran_to_value(value):
    if value == '.true.':
        return True
    elif value == '.false.':
        return False

    try:
        value = int(value)
    except(ValueError):
        try:
            value = float(value)
        except(ValueError):
            return value.strip("'")

    return value


class Singleton(type):

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in Singleton._instances:
            Singleton._instances[cls] = type.__call__(cls, *args, **kwargs)
        return Singleton._instances[cls]

    def __erase(self):
        """Reset the internal state, mainly for testing purposes"""

        Singleton._instances = {}


class SiteConfig(with_metaclass(Singleton, object)):
    """Site configuration holding details about the execution environment
    with methods for retrieving the details from systems variables and
    creating directories

    Args:
        scheduler (str) :
            Name of the scheduler, curretly supports only `SLURM` and
            `PBS`/`TORQUE`
        scratchenv (str) :
            Name of the envoronmental variable that defines the scratch path
    """

    def __init__(self, scheduler=None, cluster=None, scratchenv=None):
        self.scheduler = scheduler
        self.cluster = cluster
        self.scratchenv = scratchenv
        self.batchmode = False
        self.global_scratch = None
        self.submitdir = None
        self.scratch = None
        self.jobid = None
        self.nnodes = None
        self.nodelist = None
        self.nprocs = None

        self.set_variables()

    def set_variables(self):
        """Resolve the site attributes based on the scheduler used"""
        if self.scheduler is None:
            self.submitdir = Path(os.path.abspath(os.getcwd()))
            self.set_global_scratch(self.submitdir)
            self.jobid = os.getpid()
            return

        self.batchmode = True
        self.set_global_scratch()
        lsheduler = self.scheduler.lower()
        if lsheduler == 'slurm':
            self.set_slurm_env()
        elif lsheduler in ['pbs', 'torque']:
            self.set_pbs_env()
        elif lsheduler == 'lbs':
            self.set_lbs_env()

    @classmethod
    def check_scheduler(cls):
        """Check for one of the supported schedulers."""
        check_shedulers = {
            'SLURM': 'SLURM_CLUSTER_NAME',
            'PBS': 'PBS_SERVER',
            'LBS': 'LSB_EXEC_CLUSTER'
        }

        # Check for scheduler environment variables
        scheduler = None
        for sched, ev in check_shedulers.items():
            cluster = os.getenv(ev)
            if cluster:
                scheduler = sched
                break

        return cls(scheduler, cluster)

    def set_global_scratch(self, scratchdir=None):
        """Set the global scratch attribute"""
        if isinstance(scratchdir, str):
            self.global_scratch = Path(scratchdir)

        scratch = os.environ.get(self.scratchenv)
        if scratch is None:
            scratch = self.submitdir
        if not os.path.exists(scratch):
            raise OSError(
                '$SCRATCH variable {} points '
                'to non-existent path'.format(self.scratchenv))

        self.global_scratch = Path(scratch)

    def set_slurm_env(self):
        """Set the attributes necessary to run the job based on the
        enviromental variables associated with SLURM scheduler
        """
        self.jobid = os.getenv('SLURM_JOB_ID')
        self.submitdir = Path(os.getenv('SLURM_SUBMIT_DIR'))

        self.nnodes = int(os.getenv('SLURM_JOB_NUM_NODES'))
        tpn = int(os.getenv('SLURM_TASKS_PER_NODE').split('(')[0])
        self.nodelist = hostlist.expand_hostlist(
            os.getenv('SLURM_JOB_NODELIST'))

        proclist = list(
            itertools.chain.from_iterable(itertools.repeat(x, tpn)
                                          for x in self.nodelist))

        self.nprocs = len(proclist)

    def set_lbs_env(self):
        """Set the attributes necessary to run the job based on the
        enviromental variables associated with LBS scheduler
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
        enviromental variables associated with PBS/TORQUE scheduler
        """
        self.jobid = os.getenv('PBS_JOBID')
        self.submitdir = Path(os.getenv('PBS_O_WORKDIR'))

        nodefile = os.getenv('PBS_NODEFILE')
        with open(nodefile, 'r') as f:
            procs = [_.strip() for _ in f.readlines()]

        self.nodelist = list(set(procs))
        self.nnodes = len(self.nodelist)
        self.nprocs = len(procs)

    def make_scratch(self):
        """Create a user scratch dir on each node (in the global scratch
        area) in batchmode or a single local scratch directory otherwise
        """
        prefix = '_'.join(['qe', str(os.getuid()), str(self.jobid)])
        scratch = Path(tempfile.mkdtemp(
            prefix=prefix,
            suffix='_scratch',
            dir=self.global_scratch)).abspath()

        with cd(self.submitdir):
            if self.batchmode:
                cmd = self.get_exe_command('mkdir -p {}'.format(scratch))
                subprocess.call(cmd)
            else:
                scratch.makedirs_p()

        self.scratch = scratch
        atexit.register(self.clean)

    def get_exe_command(self, program, workdir=None):
        """Return a command as list to execute `program` through
        a supplied executable. If a workdir is provided, assume
        execution per processor, otherwise, per host.
        """
        if self.cluster == 'edison':
            exe, host, nproc, wd = 'srun', '-w', '-n', '-D'
        else:
            exe, host, nproc, wd = 'mpirun', '-host', '-np', '-wdir'

        if workdir is not None:
            command = [exe, wd, workdir]
        else:
            command = [exe, nproc, self.nnodes, host, ','.join(self.nodelist)]
        command += program.split()

        return command

    def run(self, exe='pw.x', infile='pw.pwi', outfile='pw.pwo'):
        """Run an Espresso executable."""
        mypath = os.path.abspath(os.path.dirname(__file__))
        exedir = subprocess.check_output(['which', exe]).decode()

        title = '{:<14}: {}\n{:<14}: {}'.format(
            'python dir', mypath, 'espresso dir', exedir)

        # Copy the input file to the scratch directory.
        inp = self.submitdir.joinpath(infile)
        inp.copy(self.scratch)

        # This will remove the old output file.
        output = self.submitdir.joinpath(outfile)
        with open(output, 'w') as f:
            f.write(title)

        if self.batchmode:
            parflags = ''
            if self.nprocs > 1:
                parflags += '-npool {}'.format(self.nprocs)
            command = self.get_exe_command(
                '{} {} -in {}'.format(exe, parflags, infile), self.scratch)
        else:
            command = [exe, '-in', infile]

        with cd(self.scratch):
            with open(output, 'ab') as f:
                state = subprocess.call(command, stdout=f)

        if state != 0:
            if grepy(outfile, 'JOB DONE.'):
                pass
            else:
                raise RuntimeError('Execution returned a non-zero state')

        return state

    def clean(self):
        """Remove the temporary files and directories."""
        os.chdir(self.submitdir)

        calc = self.scratch.joinpath('calc.save')
        save = self.submitdir.joinpath('calc.tar.gz')

        if os.path.exists(calc) and not save.exists():
            with tarfile.open(save, 'w:gz') as f:
                f.add(calc, arcname=calc.basename())
