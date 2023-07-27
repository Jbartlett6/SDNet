import torch 
import scipy.special 
import math

def construct_sh_basis(bvecs_sph, order):
    '''
    Takes the bvector gradient directions and the spherical harmonic coefficients max
    order and creates a matrix containing said spherical harmonic bases. Has the same 
    number of rows as there are signals (or bvectors) and the same number of columns as 
    there are spherical harmonic bases.
    '''
    A_cumulative = torch.zeros((bvecs_sph.shape[0],1)).float()
    for l in range(order+1):
        if l%2 == 0:
            A_temp = torch.zeros((bvecs_sph.shape[0],2*l+1))
            for k in range(2*l+1):
                for n in range(bvecs_sph.shape[0]):
                    A_temp[n,k] = real_sh_eval(k-l,l,bvecs_sph[n,0],bvecs_sph[n,1])
            A_cumulative = torch.cat((A_cumulative,A_temp),1)
    return A_cumulative[:,1:]

def real_sh_eval(m,n,az,el):
    '''
    Evaluates the real spherical harmonic coefficients for m = order/phase, n = degree.
    '''
    if n%2:
        sh = 0
    elif m<0:
        if m%2 == True:
            pow = 0
        else:
            pow = 1
            

        sh = ((-1)**pow)*math.sqrt(2)*torch.imag(scipy.special.sph_harm(m, n, az, el))
    elif m == 0:
        sh = scipy.special.sph_harm(m, n, az, el)
    elif m>0:
        sh = math.sqrt(2)*torch.real(scipy.special.sph_harm(m, n, az, el))
    
    return sh

