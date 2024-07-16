import numpy as np

class Population:
    """
    A class to generate and evolve a lattice according to the rules of
    Greenberg-Hastings cellular automata for diffusion on excitable media.

    ...

    Attributes
    ----------
    act : int
        <desc>
    pas : int
        <desc>
    tau0 : int
        <desc>
    r : float
        <desc>
    size : int
        <desc>
    i_0 : float
        <desc>
    r_0 : float
        <desc>
    loc : float
        <desc>
    (s,i,r)count : float
        <desc>
    (s,i,r)time : list
        <desc>
    periodic : bool
        <desc>
    p : numpy.ndarray
        <desc>

    Methods
    -------
    count()
        <desc>
    nbr(i,j)
        <desc>
    check(loc)
        <desc>
    infect(loc)
        <desc>
        

    """
    def __init__(self,act=1,pas=1,size=1,i_0=0.0,r_0=None,periodic=False,p=None):
        """
        Parameters
        ----------
            act : int
                <desc>
            pas : int
                <desc>
            r : float
                <desc>
            size : int
                <desc>
            i_0 : float
                <desc>
            r_0 : float
                <desc>
            periodic : bool
                <desc>
            p : numpy.ndarray
                <desc>

        """
        self.act=act
        self.pas=pas
        self.tau0=self.act+self.pas

        self.loc = []

        if p is None:
            self.size = size
            self.i_0 = i_0

            if r_0 is None:
                self.r_0 = 0.5
            else:
                self.r_0 = r_0

##            self.p = np.random.randint(1,self.tau0+1,size=self.size*self.size)
            self.p = np.zeros(self.size*self.size,dtype=np.int8)
            endi0 = int(self.i_0*(self.size**2))
            endr0 = endi0+int((self.r_0*(1-self.i_0))*(self.size**2))

##            self.p[0:endi0] = 1
##            self.p[endi0:endr0] = self.act+1
            self.p[0:endi0] = np.random.randint(1,self.act+1)
            self.p[endi0:endr0] = np.random.randint(self.act+1,self.tau0+1)

            np.random.shuffle(self.p)
            self.p = self.p.reshape(self.size,self.size)
        else:
            self.p = p
            self.size = len(p[0])
            self.count()
            self.i_0 = self.icount/self.size**2
            self.r_0 = self.rcount/self.size**2

        self.periodic = periodic
        self.default_nbh = [np.array([x,y]) for x in range(-self.size+1,self.size)
                            for y in range(-self.size+1,self.size)
                            if np.sqrt(x**2+y**2)<=1 and [x,y]!=[0,0]]


    def nbr(self,i,j):
        """<desc>

        Parameters
        ----------
        <var> : <type>
            <desc>

        Returns
        ----------
        <var> : <type>
            <desc>
        
        Raises
        ------
        <err>
            <desc>
        """
        
        if self.periodic:
            nbh = np.array([i,j])+self.default_nbh
            c_nbrs = []
            for i in nbh%self.size:
                c_nbrs.append(self.p[tuple(i)])
        else:
            c_nbh = np.array([i,j]) + self.default_nbh
            real_nbh = []
            for i in c_nbh:
                if 0<=i[0]<self.size and 0<=i[1]<self.size:
                    real_nbh.append(i)
            c_nbrs = []
            for i in real_nbh:
                c_nbrs.append(self.p[tuple(i)])
        return c_nbrs

    def check(self,loc):
        """<desc>

        Parameters
        ----------
        <var> : <type>
            <desc>

        Returns
        ----------
        <var> : <type>
            <desc>
        
        Raises
        ------
        <err>
            <desc>
        """

        self.loc = []
        nbhood = []
        for i in range(self.size):
            for j in range(self.size):
                if self.p[i,j] == 0:
                    if self.size==1:
                        continue
                    else:
                        nbhood = self.nbr(i,j)
                    for k in nbhood:
                        if k in range(1,self.act+1):
                            self.loc.append((i,j))
                            break
        return self.loc

    def infect(self,loc):
        """<desc>

        Parameters
        ----------
        <var> : <type>
            <desc>

        Returns
        ----------
        <var> : <type>
            <desc>
        
        Raises
        ------
        <err>
            <desc>
        """

        for i in range(self.size):
            for j in range(self.size):
                if self.p[i,j] >= self.tau0:
                    self.p[i,j] = 0
                if 1<=self.p[i,j]<self.tau0:
                    self.p[i,j] += 1
                if (i,j) in loc:
                    self.p[i,j] += 1
        return
####################################################################################

def run(com):
    """<desc>

    Parameters
    ----------
    <var> : <type>
        <desc>

    Returns
    ----------
    <var> : <type>
        <desc>
    
    Raises
    ------
    <err>
        <desc>
    """
    
    for i in com:
        i.loc = i.check(i.loc)
    for i in com:
        i.infect(i.loc)
    return com

####################################################################################
####--Global Variables--####
##Size: N x N
N = 50
##Time: transient + observation
trans = 0 ; obs = 30 ; T = trans + obs
##States: active + passive + 1 (restive {0})
active = 4 ; passive = 5
############################

if __name__=='__main__':

    import os

####Test####    
##    com=[Population(act=active,pas=passive,size=100,i_0=0.1,r_0=0.3)]
##    states=np.empty((T,int(N),int(N)),dtype=np.int8)
##    for t in range(T):
##        states[t] = np.copy(com[0].p)
##        com = run(com)
############

    ics=500
    rec=np.zeros((ics,T,N,N))
    for i0 in np.arange(0,1,0.1):
        com=[]
        for i in range(ics):
            com.append(Population(act=active,pas=passive,size=N,i_0=i0,r_0=0.3))
        for t in range(T):
            com = run(com)
            ic = 0
            for i in com:
                rec[ic,t] = np.copy(i.p)
                ic += 1
        datadir = '../data';datafile='/i0-{}.npy'.format(i0)
        if not os.path.exists(datadir):
            os.makedirs(datadir)
        np.save(datadir+datafile,rec)
