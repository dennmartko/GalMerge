import numpy as np
from tqdm import tqdm, tqdm_gui
from numba import jit,njit

@njit
def leapfrog(r, F, v, SDR, SDV, dt=0.001, init=False, store="no"):
    #v is necessary for initialization after that it's not needed as the
    #function stores v_async in between calls
    if v is None and init:
        raise ValueError("v should be provided!")

    #if init is true kickstart the leapfrog algorithm
    if init:
        vnew = v + dt / 2 * F
        return r, vnew, SDR, SDV
                                
    rnew = np.empty_like(r)
    vnew = np.empty_like(v)
    rnew = r + dt * v
    vnew = v + dt * F 
    vstore = vnew - dt / 2 * F

    #store r and v
    if store == 'yes':
        SDR = np.append(SDR, rnew, axis=0)
        SDV = np.append(SDV, vstore, axis=0)
    return rnew, vnew, SDR, SDV


if __name__ == "__main__":
    SDV = np.empty((0,2))
    SDR = np.empty((0,2))

    size = 100000 * 2
    r = np.linspace(1,10,size).reshape(int(size / 2),2)
    v = np.linspace(1,10,size).reshape(int(size / 2),2)
    F = np.linspace(1,10,size).reshape(int(size / 2),2)

    # Check if we can put this under jit as well.
    for i in tqdm(range(1000)):
        if i == 0:
            r, v, SDR, SDV = leapfrog(r, F, v, SDV,SDR, dt=0.01, init=True)
        else:
            if i % 10 == 0:
                r, v, SDR, SDV = leapfrog(r, F, v, SDV, SDR, dt=0.01, store='yes')
            else:
                r, v, SDR, SDV = leapfrog(r, F, v, SDV, SDR, dt=0.01)
    print(SDR, SDV)