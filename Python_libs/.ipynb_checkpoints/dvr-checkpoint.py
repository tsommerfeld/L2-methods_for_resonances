#
# Sine-DVR: sine function-based discreet variable representation 
#


from numpy import pi
from numpy import sin
from numpy import cos
from numpy import zeros
from numpy import linspace
from numpy import copy
from numpy import diag
from numpy.linalg import eig, eigh, eigvals, eigvalsh

def DVRGrid(x0, xNplus1, n):
    """
    The complete grid consists of N+2 equidistant points.
    Since the wavefunction vanishes on the first and last point,
    the numeric grid consists of only N points.
    """
    dx = float(xNplus1 - x0) / float(int(n) + 1)
    grid = linspace(x0+dx, xNplus1-dx, n)
    return grid


def KineticEnergy(mass, x0, xNplus1, n):
    """
    the formulae for the kinetic energy matrix elements are from 
    Physics Report 324, 1 (2000) app. B 
    (note that in the paper there is sign mistake in eq. )
    """
    delta_x = float(xNplus1 - x0) / float(n + 1)
    aux1 = 0.5*pi*pi/(mass*delta_x*delta_x)
    aux2 = pi / float(n+1)
    aux3 = float((n+1)**2)
    aux4 = 2.0 * aux1 / aux3
    aux5 = 1.0/3.0 + 1.0/(6.0*aux3)

    sins = zeros(n)
    coss = zeros(n)

    T = zeros((n,n))
    
    for a in range(n):
        sins[a] = sin((a+1)*aux2) 
        coss[a] = cos((a+1)*aux2)
        T[a,a] = aux1 * (aux5 - 0.5/(aux3 * sins[a]**2))
  
    for a in range(n):
        for b in range(a):
            tab = aux4 * sins[a] * sins[b] / (coss[a]-coss[b])**2
            if (b-a) % 2 == 1:
                T[a,b] = T[b,a] = -tab
            else:
                T[a,b] = T[b,a] = tab
    return T
    

def KineticEnergy2(mass, x0, xNplus1, m):
    """
    the formulae for the kinetic energy matrix elements are from 
    Pisces CM-DVR
    """

    n = m + 1
    len = xNplus1 - x0
    f1 = pi*pi / (4.0 * mass * len * len)
    f2 = pi / (2*n)
    f3 = (2*n*n + 1.0)/3.0

    T = zeros((m,m))    

    for i in range(m):
        T[i,i] = f1 * (f3 - 1.0/(sin(pi*(i+1)/n))**2 )
        for j in range(i):
            t_ij = f1 * (1.0/(sin(f2*(j-i)))**2 - 1.0/(sin(f2*(j+i+2)))**2)
            if (i-j)%2 == 1:
                T[i,j] = T[j,i] = -t_ij
            else:
                T[i,j] = T[j,i] = t_ij


    return T
    






def DVRDiag2(n, T, V, wf=False, sym=True):
    """
    here is a "low-level" call useful for repetitive calling 
    with different potentials (computing T is the bottleneck for small n)
    input is n, T as an n-by-n matrix, and V as a list V[] 
    builds and diagonalizes the Hamilton matrix
    returned are n energies and n wavefunctions
    """
    H = copy(T) + diag(V)
    if wf:
        if sym:
            [energy, wf] = eigh(H)
        else:
            [energy, wf] = eig(H)
        return energy, wf    
    else:
        if sym:
            energy = eigvalsh(H)
        else:
            energy = eigvals(H)
        return energy


def DVRDiag(n, x0, xNplus1, mass, potential):
    """
    here is "high-level" function that handles all in one go
    create a grid, a DVR of T, and the potential energy V(x),  
    then build and diagonalize the Hamilton matrix, 
    returned are n energies, n grid points, 
    the potential V(x) at the n grid points, and the n wavefunctions
    """
    grid = DVRGrid(x0, xNplus1, n)
    H = KineticEnergy(mass, x0, xNplus1, n)
    V = zeros(n)
    for i in range(n):
        V[i] = potential(grid[i])
        H[i,i] += V[i]
    [energy, wf] = eigh(H)
    return energy, grid, V, wf


def write_matrix(file_name, H, comment="No comment provided."):
    """
    write a matrix in the readable ascii format (1st diag, then lower triangle by rows)
    this is an ancient standard format from the HD Fortran days with specific empty lines
    
    1st line: n comment, where n is the dimension of the matrix
    2nd line: empty
    n lines: i d_ii, where d_ii is the i-th diadonal element
    empty line
    n(n-1)/2 lines: upper triangle: i  j  o_ij, 
    where i>j and o_ij is an off-diagonal element    
    """
    s = H.shape
    if len(s) != 2:
        sys.exit("Error in dvr.write _matrix: 2nd arg must be a square matrix.")
    n = s[0]
    if n != s[1]:
        sys.exit("Error in dvr.write _matrix: 2nd arg must be a square matrix.")
    f = open(file_name, 'w')

    line = str(n) + '  ' + str(comment) + '\n'
    f.write(line)
    line='\n'
    f.write(line)
    
    for i in range(n):
        line = str(i+1) + '  ' + str(H[i,i]) + '\n'
        f.write(line)
    line='\n'
    f.write(line)

    for i in range(1,n):
        for j in range(i):
            line = str(i+1) + '  ' +  str(j+1) + '  ' + str(H[i,j]) + '\n'
            f.write(line)

    f.close()
    return
