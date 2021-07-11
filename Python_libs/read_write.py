import numpy as np
import sys

def write_theta_run(fname, thetas, trajectories, header=False):
    """ 
    write data created by complex scaling, a so-called theta-run 
    data file format:
    1st column theta
    then columns with ReE ImE pairs, all separated by whitespace
    input:
        the filename for the data file
        the theta values, float, n
        the trajectories, complex, n,m
    
    """
    n_theta = len(thetas)
    (n,m) = trajectories.shape
    if n_theta != n:
        sys.exit('incompatible array lengths in write_theta_run')
    tr = np.zeros((n, 2*m+1))
    tr[:,0]=thetas
    header="theta"
    for i in range(m):
        tr[:,2*i+1]=trajectories[:,i].real
        tr[:,2*i+2]=trajectories[:,i].imag
        header = header + ', ReE' + str(i+1) + ', ImE' + str(i+1)
    np.savetxt(fname, tr, fmt='%15.12f', delimiter=' ')


def read_theta_run(fname):
    """ 
    read data created by complex scaling, a so-called theta-run 
    1st column theta
    columns with ReE ImE pairs, all separated by whitespace
    input:
        the filename of the data file
    output:
        (n_thetas, n_energies), ints: number of theta values and energies 
        thetas, float: array with theta values
        es, complex: matrix with energies, es[:,j] is the jth energy(theta)
    """
    theta_run = np.loadtxt(fname)
    (n_thetas, n_energies) = theta_run.shape
    n_energies = (n_energies-1)//2
    thetas = theta_run[:,0]
    es = np.zeros((n_thetas,n_energies), complex)
    # put the complex energies together again
    for j in range(n_energies):
        es[:,j] = theta_run[:,2*j+1] + 1j*theta_run[:,2*j+2]
    return (n_thetas, n_energies), thetas, es 
