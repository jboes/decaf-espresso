from __future__ import division, print_function

import contextlib
import itertools as its
import os
import shlex
import tempfile
import functools
from subprocess import call
import re

from six import with_metaclass
import hostlist as hl
from path import Path


def grepy(filename, search, instance=-1):
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


def preserve_cwd(function):
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        cwd = os.getcwd()
        try:
            return function(*args, **kwargs)
        finally:
            os.chdir(cwd)
    return decorator


@contextlib.contextmanager
def working_directory(path):
    """A context manager which changes the working directory to the given
    path, and then changes it back to its previous value on exit.
    """
    old_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


class Singleton(type):

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in Singleton._instances:
            Singleton._instances[cls] = type.__call__(cls, *args, **kwargs)
        return Singleton._instances[cls]

    def __erase(self):
        'Reset the internal state, mainly for testing purposes'

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

    def __init__(self, scheduler=None, usehostfile=False,
                 scratchenv='SCRATCH'):

        self.global_scratch = None
        self.scheduler = scheduler
        self.scratchenv = scratchenv
        self.localtmp = None
        self.submitdir = None
        self.usehostfile = usehostfile
        self.user_scratch = None

        # default values for the attributes that will be gathered
        self.batchmode = None
        self.hosts = None
        self.jobid = None
        self.nnodes = None
        self.nodelist = None
        self.nprocs = None
        self.proclist = None
        self.tpn = None

        self.set_variables()

    def set_variables(self):
        """Resolve the site attributes based on the scheduler used"""
        if self.scheduler is None:
            self.set_interactive()
        elif self.scheduler.lower() == 'slurm':
            self.set_slurm_env()
        elif self.scheduler.lower() in ['pbs', 'torque']:
            self.set_pbs_env()

    @classmethod
    def check_scheduler(cls):
        """Check if either SLURM or PBS/TORQUE are running"""
        scheduler = None

        # check id SLURM is installed and running
        with open(os.devnull, 'w') as devnull:
            exitcode = call('scontrol version', shell=True, stderr=devnull)
            if exitcode == 0:
                scheduler = 'SLURM'

        # check if PBS/TORQUE is installed and running
        with open(os.devnull, 'w') as devnull:
            exitcode = call('ps aux | grep pbs | grep -v grep', shell=True,
                            stderr=devnull)
            if exitcode == 0:
                scheduler = 'PBS'

        return cls(scheduler)

    def set_interactive(self):
        """Set the attributes necessary for interactive runs

        - `batchmode` is False
        - `jobid` is set to the PID
        - `global_scratch` checks for scratch under `self.scratchenv` if it is
          not defined used current directory
        """
        self.scheduler = None
        self.batchmode = False
        self.submitdir = Path(os.path.abspath(os.getcwd()))
        self.jobid = os.getpid()

        if os.getenv(self.scratchenv) is not None:
            self.global_scratch = Path(os.getenv(self.scratchenv))
        else:
            self.global_scratch = self.submitdir

    def set_global_scratch(self):
        """Set the global scratch attribute"""
        scratch = os.getenv(self.scratchenv)

        if scratch is None:
            raise OSError('SHELL variable {} is undefied'.format(self.scratchenv))
        else:
            if os.path.exists(scratch):
                self.global_scratch = Path(scratch)
            else:
                raise OSError('scratch directory <{}> defined with {} does not exist'.format(
                    scratch, self.scratchenv))

    def set_slurm_env(self):
        """Set the attributes necessary to run the job based on the
        enviromental variables associated with SLURM scheduler
        """
        self.scheduler = 'slurm'
        self.batchmode = True

        self.set_global_scratch()

        self.jobid = os.getenv('SLURM_JOB_ID')
        self.submitdir = Path(os.getenv('SUBMITDIR'))

        self.nnodes = int(os.getenv('SLURM_JOB_NUM_NODES'))
        self.tpn = int(os.getenv('SLURM_TASKS_PER_NODE').split('(')[0])
        self.nodelist = hl.expand_hostlist(os.getenv('SLURM_JOB_NODELIST'))

        self.proclist = list(its.chain.from_iterable(its.repeat(x, self.tpn)
                             for x in self.nodelist))
        self.nprocs = len(self.proclist)

    def set_pbs_env(self):
        """Set the attributes necessary to run the job based on the
        enviromental variables associated with PBS/TORQUE scheduler
        """
        self.scheduler = 'pbs'
        self.batchmode = True

        self.set_global_scratch()

        self.jobid = os.getenv('PBS_JOBID')
        self.submitdir = Path(os.getenv('PBS_O_WORKDIR'))

        nodefile = os.getenv('PBS_NODEFILE')
        with open(nodefile, 'r') as nf:
            self.hosts = [x.strip() for x in nf.readlines()]

        self.nprocs = len(self.hosts)
        uniqnodes = sorted(set(self.hosts))

        self.perHostMpiExec = ['mpirun', '-host', ','.join(uniqnodes),
                               '-np', '{0:d}'.format(len(uniqnodes))]

        self.perProcMpiExec = 'mpiexec -machinefile {nf:s} -np {np:s}'.format(
            nf=nodefile, np=str(self.nprocs)) + ' -wdir {0:s} {1:s}'

    def make_localtmp(self, workdir):
        """Create a temporary local directory for the job

        Args:
            workdir (str) :
                Name of the working directory for the run
        """
        if workdir is None or len(workdir) == 0:
            self.localtmp = self.submitdir
        else:
            self.localtmp = self.submitdir.joinpath(workdir + '_' + str(self.jobid))

        self.localtmp.makedirs_p()
        return self.localtmp.abspath()

    def make_scratch(self):
        """Create a user scratch dir on each node (in the global scratch
        area) in batchmode or a single local scratch directory otherwise
        """
        prefix = '_'.join(['qe', str(os.getuid()), str(self.jobid)])
        self.user_scratch = Path(tempfile.mkdtemp(prefix=prefix,
                                                  suffix='_scratch',
                                                  dir=self.global_scratch))

        with working_directory(str(self.localtmp)):
            if self.batchmode:
                cmd = self.get_host_mpi_command('mkdir -p {}'.format(str(self.user_scratch)))
                call(cmd)
            else:
                self.user_scratch.makedirs_p()

        return self.user_scratch.abspath()

    def get_hostfile(self):

        if self.localtmp is None:
            raise RuntimeError('<localtmp> is not defined yet')
        else:
            return self.localtmp.joinpath('hostfile')

    def get_host_mpi_command(self, program, aslist=True):
        """Return a command as list to execute `program` through
        MPI per host
        """
        command = 'mpirun -host {} '.format(','.join(self.nodelist)) +\
                  '-np {0:d} {1:s}'.format(self.nnodes, program)

        if aslist:
            return shlex.split(command)
        else:
            return command

    def get_proc_mpi_command(self, workdir, program, aslist=True):
        """Return a command as list to execute `program`
        through MPI per proc
        """
        if self.usehostfile:
            command = 'mpirun --hostfile {0:s} '.format(self.get_hostfile()) +\
                      '-np {0:d} '.format(self.nprocs) +                      \
                      '-wdir {0:s} {1:s}'.format(workdir, program)
            # should be logged print('Using hostfile',self.get_hostfile())
        else:
            command = 'mpirun -wdir {0:s} {1:s}'.format(workdir, program)
            # should be logged print('Not Using hostfile', self.get_hostfile())

        if aslist:
            return shlex.split(command)
        else:
            return command

    def write_local_hostfile(self):
        """Write the local hostfile"""
        with open(self.get_hostfile(), 'w') as fobj:
            for proc in self.proclist:
                print(proc, file=fobj)

    def __repr__(self):
        return "%s(\n%s)" % (
            (self.__class__.__name__),
            ' '.join(["\t%s=%r,\n" % (key, getattr(self, key))
                      for key in sorted(self.__dict__.keys())
                      if not key.startswith('_')]))
