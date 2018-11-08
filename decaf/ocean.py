import numpy as np
import shutil
import ase
import sys
import os
Bohr = ase.units.create_units('2006')['Bohr']

ocean_templates = {8:
    ['2\n0.3 2.00 0.0001\n3.5\n0.05 20\n',
     '008\n1 0 0 0\nscalar rel\nlda\n2.0 3.8 0.0 0.0\n2.0 3.8 0.0 0.0\n']
}

default_run_parameters = {
    'ppdir': ['../'],
    'dft': 'qe',
    'control': 0,
    'ser_prefix': ['srun -n 1'],
}

default_calculator_parameters = {
    'dft.startingwfc': ['atomic+random'],
    'dft.diagonalization': ['david'],
    'dft_energy_range': 50,
    'nkpt': [6, 6, 6],
    'ngkpt': [6, 6, 6],
    'screen.nkpt': [2, 2, 2],
    'screen.nbands': 2000,
    'ecut': 120,
    'core_offset': True,
    'metal': True,
    'occopt': 3,
    'degauss': 0.004,
    'fband': 0.65,
    'toldfe': 1.7e-6,
    'tolwfr': 1.1e-16,
    'nstep': 600,
    'mixing': 0.1,
    'mixing_ndim': 20,
    'etol': 1.5e-7,
}

default_ocean_parameters = {
    'diemac': 30,
    'nedges': 1,
    'edges': [-8, 1, 0],
    'screen.shells': [6.0],
    'cnbse.rad': [6.0],
    'cnbse.broaden': [0.3],
    'cnbse.niter': 200,
    'scfac': 0.80,
}

class Ocean():
    """Ocean code interface

    More to come once I know more.
    """

    def __init__(self, atoms, nodes=6, **kwargs):
        self.atoms = atoms
        self.nodes = nodes

        self.run_parameters = default_run_parameters.copy()
        self.run_parameters['para_prefix'] = ['srun -n {}'.format(32 * nodes)]
        for k, v in default_run_parameters.items():
            user_defined = kwargs.get(k)
            if user_defined:
                self.run_parameters[k] = user_defined

        self.calculator_parameters = default_calculator_parameters.copy()
        for par in default_calculator_parameters.items():
            user_defined = kwargs.get(k)
            if user_defined:
                self.calculator_parameters[k] = user_defined

        self.ocean_parameters = default_ocean_parameters.copy()
        for par in default_ocean_parameters.items():
            user_defined = kwargs.get(k)
            if user_defined:
                self.ocean_parameters[k] = user_defined
        

    def write_input(self):
        unique_numbers, atom_indicies = np.unique(
            self.atoms.numbers, return_inverse=True)
        atom_indicies += 1
        str_unique_numbers = ' '.join(unique_numbers.astype(str))
        str_atom_indicies = ' '.join(atom_indicies.astype(str))

        positions = self.atoms.get_scaled_positions()

        str_cell = ''.join(['\n'.join(
            '   {:<0.5f} {:<0.5f} {:<0.5f}'.format(*p)
            for p in self.atoms.cell)])

        str_positions = ''.join(['\n'.join(
            '   {:<0.5f} {:<0.5f} {:<0.5f}'.format(*p)
            for p in positions)])

        # Parse the atoms object an write PP and Ocean files
        ocean_fill, ocean_opts, pp_list = '', '', ''
        for num in unique_numbers:
            sym = ase.data.chemical_symbols[num]

            ppfile = '{:02d}-{}.GGA.fhi'.format(num, sym)
            pp_list += "\n   {}".format(ppfile)

            cwd = '/'.join(sys.modules[__name__].__file__.split('/')[:-2])
            shutil.copy('{}/gbrv15pbe/{}.UPF'.format(cwd, sym), ppfile)

            template = ocean_templates.get(num)
            if template:
                # Needs an ocean file
                fill_file = '{0}.fill'.format(sym)
                opts_file = '{0}.opts'.format(sym)
                ocean_fill = '   {0} {1}\n'.format(num, fill_file)
                ocean_opts = '   {0} {1}\n'.format(num, opts_file)

                with open(opts_file, 'w') as f:
                    f.write(template[1])

                with open(fill_file, 'w') as f:
                    f.write(template[0])

        ocean_plate = ''
        if ocean_fill:
            ocean_plate += 'opf.fill {{\n{0}}}\n'.format(ocean_fill)
            ocean_plate += 'opf.fill {{\n{0}}}\n'.format(ocean_opts)

        run_plate = ''
        for k, v in self.run_parameters.items():
            if isinstance(v, list):
                v = '{' + ' '.join([str(p) for p in v]) + '}'
            else:
                v = str(v)
            run_plate += k + ' ' + v + '\n'

        calculator_plate = ''
        for k, v in self.calculator_parameters.items():
            if isinstance(v, list):
                v = '{' + ' '.join([str(p) for p in v]) + '}'
            elif isinstance(v, bool):
                v = '.{}.'.format(str(v).lower())
            else:
                v = str(v)
            calculator_plate += k + ' ' + v + '\n'

        for k, v in self.ocean_parameters.items():
            if isinstance(v, list):
                v = '{' + ' '.join([str(p) for p in v]) + '}'
            else:
                v = str(v)
            ocean_plate += k + ' ' + v + '\n'

        for i, eye in enumerate(np.eye(3, dtype=int)):
            eye = '  '.join(eye[::-1].astype(str))
            photon = "dipole\ncartesian\n  {}\n" \
                "endcartesian\n  0  1  0\nend\n696".format(eye)

            with open('photon{}'.format(i+1), 'w') as f:
                f.write(photon)

        atoms_plate = """acell {{{0} {0} {0}}}

rprim {{
{1}
}}

pp_list {{{2}
}}

ntypat {3}
znucl {{{4}}}

natoms {5}
typat {{{6}}}

xred {{
{7}
}}
""".format(
    Bohr**-1,
    str_cell,
    pp_list,
    len(unique_numbers),
    str_unique_numbers,
    len(self.atoms),
    str_atom_indicies,
    str_positions)

        boilerplate = '\n##################\n'.join(
            [run_plate,
             calculator_plate,
             atoms_plate,
             ocean_plate])

        # Write the input file
        with open('input.in', 'w') as f:
            f.write(boilerplate)

        run_template = """#!/bin/bash
#SBATCH -N {} -t 6:00:00 -J ./input.in
#SBATCH  -q premium  -A m2464 -C haswell
module load fftw
module unload cray-libsci
module load abinit
module load espresso
export OMP_NUM_THREADS=2

time /global/project/projectdirs/m2464/OCEAN/2.5.2/Cori/ocean.pl ./input.in
""".format(self.nodes)
        
        with open('run.sh', 'w') as f:
            f.write(run_template)
        os.chmod('input.in', 755)
        os.chmod('run.sh', 755)

        return boilerplate

