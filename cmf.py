import numpy as np
import pandas
import os
from matplotlib import pyplot  as plt
import pickle
import scipy.interpolate as interpolate

class HILevelPopulations:
    '''
    Compute level population for HI using the cascade matrix formalism.
    See Osterbrock & Ferland 2006, section 4.2
    '''
    def __init__(self, nmax=60, TabulatedEinsteinAs = '/cosma/home/dphlss/tt/Codes/EinsteinAs/EinsteinA.dat',
                TabulatedRecombinationRates = '/cosma/home/dphlss/tt/Data/Recomb/h_iso_recomb.dat', 
                caseB = True, caseBnmax=5, verbose=False):

        # set maximum number of principle quantum number to be used - max allowed is 40
        self.nmax = nmax
        if self.nmax > 100:
            self.nmax = 100
        self.verbose = verbose
        self.caseB  = caseB
        self.caseBnmax = caseBnmax
        assert self.caseBnmax <= self.nmax, 'Please check input, caseBnmax should be smaller then nmax. '
        
        # naming convention for configuration level l
        # standard naming convention
        import string
        lforms = np.asarray(['s', 'p', 'd', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n'])
        if nmax >= len(lforms):
            def make_lforms():
                lforms = np.concatenate((list(string.ascii_lowercase), list(string.ascii_uppercase)))
                aa     = []
                for a,b in zip(string.ascii_lowercase, string.ascii_lowercase):
                    aa.append(a+b)
                AA     = []
                for A,B in zip(string.ascii_uppercase, string.ascii_uppercase):
                    AA.append(A+B)
                lforms = np.concatenate((lforms, list(aa), list(AA)))
                return lforms
            lforms = make_lforms()
        self.lforms = lforms
        
        # Read Einstein coefficients
        self.TabulatedEinsteinAs = TabulatedEinsteinAs  # name of the file
        self.A                   = self.ReadTabulatedEinsteinCoefficients(TabulatedEinsteinAs)

    
        # Read level-resolved recombination rates
        self.TabulatedRecombinationRates = TabulatedRecombinationRates                   # name of the file
        self.Recom_table = self.ReadRecombinationRates(self.TabulatedRecombinationRates) # tabulated rates
        self.Alpha_nl = self.FitRecombinationRates()                                     # fitting function to tabulated rates
        if verbose:
            print("Recombination rates read and fitted")

        # Compute cascade matrix
        self.C    = self.ComputeCascadeMatrix()
        if verbose:
            print("Cascade matrix class initialized ")
       
        
    ##################################################################    
    #                Recombination rate method                       #
    ##################################################################  
    def AlphaA(self, LogT=4.0):
        ''' Fit to case-A recombination coefficient at log temperature LogT'''
        T      = 10.**LogT
        lamb   = 315614 / T
        alphaA = 1.269e-13 * lamb**(1.503)*(1.0+(lamb/0.522)**(0.470))**(-1.923)
        return alphaA

    def AlphaB(self, LogT=4.0):
        ''' Fit to case-B recombination coefficient at log temperature T'''        
        T      = 10.**LogT
        lamb   = 315614 / T
        alphaB = 2.753e-14 * lamb**(1.5)*(1.0+(lamb/2.740)**(0.470))**(-2.2324)
        return alphaB

    def PlotRecombinationRates(self, fontsize=20, fname='recom_rate.png', plotfit=False):
        '''
        Plot examples of level-resolved recombination rate. Save result in file fname
        '''
        # get tabulated recombination rates
        rates      = self.ReadRecombinationRates(self.TabulatedRecombinationRates)
        LogTs      = rates['LogTs']
        recom_data = rates['recom_data']    

        # create figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        colors     = ['red', 'blue', 'green', 'cyan', 'orange', 'purple']
        linestyles = ['solid', 'dotted', 'dashed', 'dashdot']
        markers    = ['o', '<', 'D', '+']

        #
        index    = 1
        verbose  = False
        LogTemps = np.arange(3, 6, 0.1)
        labels  = []
        labels2 = []
        
        #
        for n, color in zip(np.arange(1, 5), colors):
            for l, linestyle, marker in zip(np.arange(n), linestyles, markers):
                conf      = HI.Config(n=n, l=l)
                index     = int((n-1)*n/2) + l
                label     = '{}'.format(conf)
                
                # table data
                h, = ax.plot(LogTs, recom_data.iloc[index, 2: 43].values, linestyle=linestyle, color=color, label=label)
                labels.append(copy.copy(h))
                
                # overplot fit
                if plotfit:
                    rr    = HI.Alpha_nl[conf]
                    rate  = rr(LogTemps)
                    h, = ax.plot(LogTemps, rate, marker=marker, linestyle='', color=color, label=label)
                    labels2.append(copy.copy(h))

        legend = ax.legend(handles=labels, loc='upper center', ncol=2, frameon=False, fontsize=0.8*fontsize, labelcolor='linecolor')
        ax.add_artist(legend)
        legend = ax.legend(handles=labels2, loc='lower left', ncol=2, frameon=False, fontsize=0.8*fontsize, labelcolor='linecolor')
        ax.add_artist(legend)
        
        #
        ax.set_xlabel(r'log $T$ [K]')
        ax.set_ylabel('log Recombination rate coefficient [cm$^{3}$ / s]')
        ax.set_xlim(3.3, 5.9)
        ax.set_ylim(-15.8, -11.4)
        # annotations
        ax.text(3.7, -11.7, "Table", va="center", ha="center", fontsize=0.8*fontsize)
        ax.text(3.7, -14.7, "Interpolated", va="center", ha="center", fontsize=0.8*fontsize)
        
        
        fig.patch.set_facecolor('white')
        fig.savefig('./recom_rate.png', bbox_inches='tight')
        fig.show()        
        
    def ReadRecombinationRates(self, fname):
        '''
        Read level-resolved recombination rates from ascii file fname, and return them
        '''
        
        # contents of the cloudy data file containing l-resolved recombination rates
        # the first line is a comment
        # the second line is the total recombination rate (case-A value)
        # the next lines give the l-resolved recombination rates
        # line = 3: n=1, l=0
        # line = 4: n=2, l=0
        # line = 5: n=2, l=1
        # etc
        # the pandas dataframe recom_data ignores the first line, so that line 1 is case A, line 2 is nl=(1,0), etc
        verbose = False   # set to true to get more information
        
        import pandas
        temp_index = np.arange(41)
        temp_index = [str(x) for x in temp_index]
        # level n has n-1 l values, so number total nyumber of resolved levels is nmax*(nmax+1)/2
        #   first row is a magic number, and we start from 0 - hence an offset of 2
        nrows      = int(self.nmax * (self.nmax+1) / 2) + 2
        rows       = np.arange(1, nrows)
        colnames   = ['Z', 'levels'] + temp_index
        try:
            recom_data = pandas.read_csv(fname, delimiter='\t', names=colnames, skiprows=lambda x: x not in rows)
            if verbose:
                print("Successfully read {} l-resolved levels ".format(self.nmax))
        except:
            print("Error reading recombination rates ")
            return -1
        LogTs      = np.linspace(0, 10, 41, endpoint=True)
        return {'LogTs':LogTs, 'recom_data':recom_data}
        

    def FitRecombinationRates(self):
        ''' 
        Provide fitting function for recombination rate as a function of Log T
        
        The fitting funcition is of the form 
        
        Recombination_rate(n, l, T) = 10**Alpha_nk(10**LogT)
        '''
        def get_l_level_index(n=1, l=0):
                assert type(n) == np.int64 and type(l) == np.int64, 'n and l must be intergers.'
                assert n >= 1, 'Principle quantum number can not be smaller than 1.'
                assert l < n and l >= 0, 'Angular momentum must be positive and smalled than principle quantum number.'

                # index is numbered from 1 to number of levels up to (nl)
                # offset by 1, since first line is a comment line
                return int((n-1)*n/2) + l # + 1
            
        def FitRate(recom_data, LogTs, n=1, l=0):
            index   = get_l_level_index(n=n, l=l)
            rate    = interpolate.interp1d(LogTs, recom_data.iloc[index, 2:43].values,fill_value="extrapolate", bounds_error=False)   
            return rate

        rates      = self.ReadRecombinationRates(self.TabulatedRecombinationRates)
        LogTs      = rates['LogTs']
        recom_data = rates['recom_data']
        
        #
        nmax     = self.nmax
        Alpha_nl = {}
        for n in np.arange(1, nmax+1):
            for l in np.arange(n):
                conf_i             = self.Config(n=n, l=l)
                Alpha_nl[conf_i]   = FitRate(recom_data, LogTs, n=n, l=l)
        return Alpha_nl
    
    
    
    ##################################################################    
    #                Cascade matrix methods                          #
    ##################################################################
    def TestAllLevelPops(self, nH = 1.0, ne = 1.0, LogT = 4.0, N={}):
        '''
        Verify whether these pop levels satisfy the equilibrium relation - Eq. 4.1
        '''
        #
        nmax     = self.nmax
        Config   = self.Config
        Alpha_nl = self.Alpha_nl
        A        = self.A
        #
        TestConfig = []
        TestDiff   = []
        for n in np.arange(1, nmax+1):
            for l in np.arange(n):
                lhs    = 0.0
                conf   = Config(n=n, l=l)
                lhs   += nH * ne * 10**Alpha_nl[conf](LogT)
                #
                for nu in np.arange(n+1, nmax+1):
                    for lu in [l-1, l+1]:
                        if (lu>= 0) & (lu < nu):
                            conf_i = Config(n=nu, l=lu)
                            lhs += N[conf_i] * A[conf_i][conf]
                #
                rhs = 0.0
                for nd in np.arange(1, n):
                    for ld in [l-1, l+1]:
                        if (ld >= 0) & (ld < nd):
                            conf_k = Config(n=nd, l=ld)
                            rhs    += A[conf][conf_k]
                    if (nd == 1) & (n==2) & (l == 0):
                        ld      = 0
                        conf_k  = Config(n=nd, l=ld)
                        rhs    += A[conf][conf_k]
                #
                Nnl  = 0.0
                diff = 1e2
                if rhs > 0:
                    Nnl  = lhs / rhs
                    diff = (Nnl-N[conf])/N[conf] * 100.
#                     if n < 10:
#                         print("Conf = {0:s}, % diff = {1:1.4f}, N = {2:1.3e}".format(conf, diff, (N[conf])))
                TestConfig.append(conf)
                TestDiff.append(diff)
        return {'Conf':TestConfig, 'Diff':TestDiff}

    def ComputeLevelPop(self, nH = 1.0, ne = 1.0, LogT=4.0, n=2, l=0, verbose=False):
        '''
        Compute level population for a given level  - implementents Eq. 4.10
        Input: 
           nH   = proton number density nH [cm^[-3]]
           ne   = electron number density [[cm^-3]]
           logT = logarithm of temperature
           n    = principle quantum number of desired level
           l    = angular momentum state of this level
        '''

        #
        nmax     = self.nmax
        A        = self.A
        C        = self.C
        Alpha_nl = self.Alpha_nl
        Config   = self.Config

        # test for consistency
        if (n < 1) or (n > self.nmax):
            print("Error: n needs to be in range 2 - {}".format(self.nmax))
        if (l<0) or (l >= n):
            print("Error: l needs to be in the range 0 -- {}".format(n-1))
        
        #
        lhs    = np.zeros_like(LogT)
        conf_k = Config(n=n, l=l)
        for nu in np.arange(n, nmax+1):
            for lu in np.arange(nu):
                conf_i = Config(n=nu, l=lu)
                lhs   += 10**Alpha_nl[conf_i](LogT) * C[conf_i][conf_k]
        lhs *= nH * ne

        # 
        rhs    = np.zeros_like(LogT)
        conf_i = Config(n=n, l=l)
        for nd in np.arange(1, n):
            for ld in [l-1, l+1]:
                if (ld >=0) & (ld < nd):
                    conf_k = Config(n=nd, l=ld)
                    rhs += A[conf_i][conf_k]
            if (nd == 1) & (n == 2) & (l == 0):
                ld     = 0
                conf_k = Config(n=nd, l=ld)
                rhs    += A[conf_i][conf_k]

        N       = np.zeros_like(LogT)
        mask    = rhs > 0
        N[mask] = lhs[mask]/rhs[mask]
        if verbose:
            print("Computed level pop for level = {0:s}, log N = {1:2.4f}".format(conf_i, np.log10(N)))
        return N

    def ComputeAllLevelPops(self, nH = 1.0, ne = 1.0, LogT=4.0):
        '''
        Compute level population for all levels  - implementents Eq. 4.10
        Input: 
           nH   = proton number density nH [cm^[-3]]
           ne   = electron number density [[cm^-3]]
           logT = logarithm of temperature
        '''
        
        #
        nmax     = self.nmax
        A        = self.A
        C        = self.C
        Alpha_nl = self.Alpha_nl
        Config   = self.Config
        

        #
        N        = {}
        for n in np.arange(1, nmax+1):
            for l in np.arange(n):
                lhs    = 0.0
                conf_k = Config(n=n, l=l)
                for nu in np.arange(n, nmax+1):
                    for lu in np.arange(nu):
                        conf_i = Config(n=nu, l=lu)
                        lhs   += 10**Alpha_nl[conf_i](LogT) * C[conf_i][conf_k]
                lhs *= nH * ne

                # 
                rhs    = 0.0
                conf_i = Config(n=n, l=l)
                for nd in np.arange(1, n):
                    for ld in [l-1, l+1]:
                        if (ld >=0) & (ld < nd):
                            conf_k = Config(n=nd, l=ld)
                            rhs += A[conf_i][conf_k]
                    if (nd == 1) & (n == 2) & (l == 0):
                        ld     = 0
                        conf_k = Config(n=nd, l=ld)
                        rhs    += A[conf_i][conf_k]

                N[conf_i] = 0.0
                if rhs>0:
                    N[conf_i] = lhs/rhs
        return N
        
    def ComputeCascadeMatrix(self):
        '''
           Compute cascade matrix from Einstein coefficients
        '''
        import time
        nmax     = self.nmax          # max upper level
        A        = self.A             # Einstein coefficient
        verbose  = self.verbose
        Config   = self.Config
        topickle = True
        
        # if pickle file exists, read it
        if topickle:
            import pickle
            if self.caseB:
                pname   = 'CascadeC_' + str(self.nmax) + '_' + 'B_lynmax_' + str(self.caseBnmax) + '.pickle'
            else:
                pname   = 'CascadeC_' + str(self.nmax) + '_' + 'A.pickle'
            try:
                with open(os.path.join('/cosma/home/dp004/dc-liu3/snap7/sims/cloudy/HII_region/pickle_data', pname), 'rb') as file:
                    data = pickle.load(file)

                # check if nmax is correct
                success = (self.nmax == data['nmax'])
                if success:
                    C = data['C']
                    P = data['P']
                    self.P = P
                    if self.verbose:
                        print("Cascade matrix coefficients unpickled")
                    return C
                else:
                    if self.verbose:
                        print("Computing cascade matrix coefficients")
            except:
                pass
        else:
            if self.verbose:
                 print("Computing cascade matrix coefficients")
        
        
        # compute probability matrix (eq. 4.8)
        from copy import deepcopy
        import time
        t0 = time.time()
        P  = deepcopy(A)
        if self.verbose:
            print(" ... Cascade matrix: P copied in time {0:1.2f} s".format(time.time()-t0))
        #
        t0 = time.time()
        for nu in np.arange(2, nmax+1):
            for lu in np.arange(nu):
                conf_i    = Config(n=nu, l=lu)
                #
                denom  = 0.0
                if conf_i == (2,0):
                    denom += A[(2,0)][(1,0)]
                for nprime in np.arange(nu):
                    for lprime in [lu-1, lu+1]:
                        if (lprime >= 0) & (lprime < nprime):
                            conf_prime = Config(n=nprime, l=lprime)
                            denom     += A[conf_i][conf_prime]
                for nd in np.arange(1, nu):
                    # add 2s->1s forbidden transition
                    if (nd == 1) & (nu == 2) & (lu == 0):
                        ld     = 0
                        conf_k = Config(n=nd, l=ld)
                        P[conf_i][conf_k] = 1.0
                        
                    # other transitions
                    if denom > 0:
                        for ld in [lu-1, lu+1]:
                            if (ld >= 0) & (ld < nd):
                                conf_k = Config(n=nd, l=ld)
                                P[conf_i][conf_k] = A[conf_i][conf_k] / denom
        if self.verbose:
            print(" ... Cascade matrix: probability matrix computed (eq. 4.8) in time {0:1.2f}".format(time.time()-t0))
        self.P = P
        
        # Compute the transpose of P
        t1 = time.time()
        Pt = {}
        for nd in np.arange(1, nmax+1):
            for ld in np.arange(nd):
                conf_k = Config(n=nd, l=ld)
                Pt[conf_k] = {}
                for nu in np.arange(nd+1, nmax+1):
                    for lu in np.arange(nu):
                        conf_i = Config(n=nu, l=lu)
                        Pt[conf_k][conf_i] = P[conf_i][conf_k]
        if self.verbose:
            print(" ... Cascade matrix: transpose of probability matrix computed in time {0:1.2f}".format(time.time()-t1))
                        
                
        # Compute cascade matrix (eq. 4.10)
        t1 = time.time()
        C  = {}
        for nu in np.arange(1, nmax+1):
            for lu in np.arange(nu):
                conf_i    = Config(n=nu, l=lu)
                C[conf_i] = {}
                for nd in np.arange(1, nu+1):
                    for ld in np.arange(nd):
                        conf_k = Config(n=nd, l=ld)
                        C[conf_i][conf_k] = 0.0
                        if (nd==nu) & (ld==lu):
                            C[conf_i][conf_k] = 1.0

        # Initialize recurrence (below 4.8)
        nu   = nmax
        nd   = nu - 1
        for lu in np.arange(nu):
            conf_i    = Config(n=nu, l=lu)
            for ld in [lu-1, lu+1]:
                if (ld >= 0) & (ld < nd):
                    conf_k = Config(n=nd, l=ld)
                    C[conf_i][conf_k] = P[conf_i][conf_k]
                    
        if verbose:
            print(" ... Cascade matrix: matrix initialized (eq. 4.10) in time {0:1.2f}".format(time.time()-t1))

                    
        # add 2s->1s forbidden transition
        conf_i            = Config(n=2, l=0)
        conf_k            = Config(n=1, l=0)
        C[conf_i][conf_k] = P[conf_i][conf_k]
        
        # Recur (complete Equation 4.10)
        tp2 = time.time()
        
        #
        for nu in np.arange(nmax, 0, -1):
            tsplit = time.time()
            for lu in np.arange(nu):
                conf_i    = Config(n=nu, l=lu)
                #
                for nd in np.arange(nu, 0, -1):
                    for ld in np.arange(nd):
                        conf_k = Config(nd, ld)
                       # create list, conf_prime, of all intermediate levels that contribute
                        conf_prime = []
                        C_prime    = []
                        P_prime    = []
                        for lprime in [ld-1, ld+1]:
                            if (lprime >=0) & (lprime < nd+1):
                                for nprime in range(nd+1, nu+1):
                                    conf = Config(nprime, lprime)
                                    C_prime.append(C[conf_i][conf])
                                    P_prime.append(Pt[conf_k][conf])
                        res = np.sum(np.array(C_prime) * np.array(P_prime))

                        # update cascade matrix
                        C[conf_i][conf_k] += res
            tsplit = time.time() - tsplit
            print(" ...    Computed level = {0:d} in time {1:1.2f}, len = {2:d}".format(nu, tsplit, len(C_prime)))
        tp2  = time.time() - tp2

        if verbose:
            print(" ... Cascade matrix: calculation finished in time {0:1.2f}s".format(tp2))
            
        # save as a pickle file
        if topickle:
            data = {'nmax':self.nmax, 'C':C, 'P':P}
            with open(os.path.join('/cosma/home/dp004/dc-liu3/snap7/sims/cloudy/HII_region/pickle_data', pname), 'wb') as file:
                pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
            if self.verbose:
                print("Cascade matric elements pickled to file ", pname)


        return C

                    
    def ReadTabulatedEinsteinCoefficients(self, fname):
        '''
          Read tabulated Einstein coefficients
          Use these to compute the casecade
        '''
        
        verbose = False  # set tru to get timing info
        
        # if pickle file exists, read it
        import pickle
        pname   = 'EinsteinA.pickle'
        try:
            with open(pname, 'rb') as file:
                data = pickle.load(file)

            # check if nmax is correct
            if caseB == True:
                success = ( (self.nmax == data['nmax']) & (self.caseBnmax == data['lynmax'] == self.caseBnmax) & (data['caseflag'] == 'B') )
            else: # case A
                success = ( (self.nmax == data['nmax']) & (data['caseflag'] == 'A') )
            if success:
                A = data['A']
                if self.verbose:
                    print("Einstein coefficients unpickled")
                return A
            else:
                if self.verbose:
                    print("Reading Einstein coefficients from file {}".format(fname))
        except:
            pass

        # Nist value of forbidden 2s-1s transition. This value is not in the data file read here
        A_2s_1s = 8.224 #2.496e-06
        
        
        # columns are n_low, l_low, n_up, l_up, A [1/s]
        import time
        tinit    = time.time()
        dtype    = {'names': ('nd', 'ld', 'nu', 'lu', 'A'),
                  'formats': (np.int32, np.int32, np.int32, np.int32, np.float64)}
        data    = np.loadtxt(fname, delimiter=",", dtype=dtype, comments='#', ).T
        nmax    = self.nmax
        tinit   = time.time() - tinit
        if self.verbose:
            print(" ... Read numerical data in time {0:1.2f}".format(tinit))

        # create Einstein coefficients dictionary
        t0      = time.time()
        A       = {}
        # loop over upper level
        for nu in np.arange(2, nmax+1):
            for lu in np.arange(nu):
                conf_i = self.Config(n=nu, l=lu)
                A[conf_i] = {}
                # loop over lower level
                for nd in np.arange(nu):
                    for ld in np.arange(nd):
                        conf_k = self.Config(n=nd, l=ld)
                        A[conf_i][conf_k] = 0
        t0 = time.time() - t0
        if verbose:
            print(" ... Created dictionary of Einstein coefficients in a time {0:1.2f}".format(t0))
                        
        # insert the values from the file
        t1       = time.time()
        nups     = data['nu'][:]
        lups     = data['lu'][:]
        nds      = data['nd'][:]
        lds      = data['ld'][:]
        Avals    = data['A'][:]
        for nup, lup, nd, ld, Aval in zip(nups, lups, nds, lds, Avals):
            conf_i = self.Config(n=nup, l=lup)
            conf_k = self.Config(n=nd, l=ld)
            if nup <= nmax:
                A[conf_i][conf_k] = Aval
            else:
                continue
        t1 = time.time() - t1
        if verbose:
            print(" ... Inserted numerical values in Einstein dictionary in a time {0:1.2}".format(t1))
        
        # insert A_2s-1s
        nu = 2
        lu = 0
        nd = 1
        ld = 0
        conf_i = self.Config(n=nu, l=lu)
        conf_k = self.Config(n=nd, l=ld)
        A[conf_i][conf_k] = A_2s_1s

        # caseA or caseB?
        if self.caseB:
            if self.verbose:
                print(" ... Imposing caseB (no Lyman-transitions) ")
            conf_k = self.Config(n=1, l=0) # ground state
            for nu in np.arange(2, self.caseBnmax+1):
                conf_i = self.Config(n=nu, l=1) # p-state
                A[conf_i][conf_k] = 0.0
        

        # save as a pickle file
        if self.caseB == False:
            caseflag = 'A'
            data = {'nmax':self.nmax, 'case': caseflag, 'A':A}
        else:
            caseflag = 'B'
            data = {'nmax':self.nmax, 'lynmax':self.caseBnmax, 'case': caseflag, 'A':A}
        with open(pname, 'wb') as file:
            pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
        if self.verbose:
            print(" ... Einstein dictionary pickled to file {}".format(pname))
        return A
    
        
    def Config(self, n=1, l=1):
        '''
              configuration states are tuples of the form (n,l), where:
          n = principle quantum number, n=1->nmax
          l = angular momentum number, l=0->n-1
        '''
        return (n,l)
        
    def DeConfig(self, config='1s'):
        '''
            extract n and l value for a given configuration state
        '''
        return config[0], config[1]


