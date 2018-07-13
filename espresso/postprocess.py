import os
import numpy as np


class XEspresso():

    def calc_pdos(self,
        Emin = None,
        Emax = None,
        DeltaE = None,
        nscf = False,
        tetrahedra = False,
        slab = False,
        kpts = None,
        kptshift = None,
        nbands = None,
        ngauss = None,
        sigma = None,
        nscf_fermilevel=False,
        add_higher_channels=False):
        """
        Calculate (projected) density of states.
        - Emin,Emax,DeltaE define the energy window.
        - nscf=True will cause a non-selfconsistent calculation to be performed
          on top of a previous converged scf calculation, with the advantage
          that more kpts and more nbands can be defined improving the quality/
          increasing the energy range of the DOS.
        - tetrahedra=True (in addition to nscf=True) means use tetrahedron
          (i.e. smearing-free) method for DOS
        - slab=True: use triangle method insead of tetrahedron method
          (for 2D system perp. to z-direction)
        - sigma != None sets/overrides the smearing to calculate the DOS
          (also overrides tetrahedron/triangle settings)
        - get_overlap_integrals=True: also return k-point- and band-resolved
          projections (which are summed up and smeared to obtain the PDOS)
        Returns an array containing the energy window,
        the DOS over the same range,
        and the PDOS as an array (index: atom number 0..n-1) of dictionaries.
        The dictionary keys are the angular momentum channels 's','p','d'...
        (or e.g. 'p,j=0.5', 'p,j=1.5' in the case of LS-coupling).
        Each dictionary contains an array of arrays of the total and
        m-resolved PDOS over the energy window.
        In case of spin-polarization, total up is followed by total down, by
        first m with spin up, etc...
        """
        efermi = self.get_fermi_level()

        # run a nscf calculation with e.g. tetrahedra or more k-points etc.
        if nscf:
            if not hasattr(self, 'natoms'):
                self.atoms2species()
                self.natoms = len(self.atoms)
            self.write_input(filename='pwnscf.inp',
                             calculation='nscf', overridekpts=kpts,
                             overridekptshift=kptshift,
                             overridenbands=nbands)
            self.run('pw.x', 'pwnscf.inp', 'pwnscf.log')
            if nscf_fermilevel:
                p = os.popen('grep Fermi '+self.localtmp+'/pwnscf.log|tail -1', 'r')
                efermi = float(p.readline().split()[-2])
                p.close()

        # remove old wave function projections
        os.system('rm -f '+self.scratch+'/*_wfc*')
        # create input for projwfc.x
        fpdos = open(self.localtmp+'/pdos.inp', 'w')
        print('&PROJWFC\n  prefix=\'calc\',\n  outdir=\'.\',', file=fpdos)
        if Emin is not None:
            print('  Emin = '+num2str(Emin+efermi)+',', file=fpdos)
        if Emax is not None:
            print('  Emax = '+num2str(Emax+efermi)+',', file=fpdos)
        if DeltaE is not None:
            print('  DeltaE = '+num2str(DeltaE)+',', file=fpdos)
        if ngauss is not None:
            print('  ngauss = '+str(ngauss)+',', file=fpdos)
        if sigma is not None:
            print('  degauss = '+num2str(sigma/Rydberg)+',', file=fpdos)
        print('/', file=fpdos)
        fpdos.close()
        # run projwfc.x
        self.site.run('projwfc.x', 'pdos.inp', 'pdos.log')


        # read in total density of states
        dos = np.loadtxt(self.scratch + '/calc.pdos_tot')
        if len(dos[0]) > 3:
            nspin = 2
            self.dos_total = [dos[:, 1], dos[:, 2]]
        else:
            nspin = 1
            self.dos_total = dos[:,1]
        self.dos_energies = dos[:,0] - efermi
        npoints = len(self.dos_energies)

        channels = {'s':0, 'p':1, 'd':2, 'f':3}
        # read in projections onto atomic orbitals
        self.pdos = [{} for i in range(self.natoms)]
        p = os.popen('ls '+self.scratch+'/calc.pdos_atm*')
        proj = p.readlines()
        p.close()
        proj.sort()
        for i, inp in enumerate(proj):
            inpfile = inp.strip()
            pdosinp = np.genfromtxt(inpfile)
            spl = inpfile.split('#')
            iatom = int(spl[1].split('(')[0])-1
            channel = spl[2].split('(')[1].rstrip(')').replace('_j',',j=')
            jpos = channel.find('j=')
            if jpos<0:
                #ncomponents = 2*l+1 +1  (latter for m summed up)
                ncomponents = (2*channels[channel[0]]+2) * nspin
            else:
                #ncomponents = 2*j+1 +1  (latter for m summed up)
                ncomponents = int(2.*float(channel[jpos+2:]))+2
            if channel not in list(self.pdos[iatom].keys()):
                self.pdos[iatom][channel] = np.zeros((ncomponents,npoints), np.float)
                first = True
            else:
                first = False
            if add_higher_channels or first:
                for j in range(ncomponents):
                    self.pdos[iatom][channel][j] += pdosinp[:,(j+1)]

        return self.dos_energies, self.dos_total, self.pdos
