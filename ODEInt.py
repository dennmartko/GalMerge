import numpy as np
from tqdm import tqdm, tqdm_gui
from numba import jit,njit

@njit
def leapfrog(r, F, v, dt=0.001, init=False):
    #v is necessary for initialization after that it's not needed as the
    #function stores v_async in between calls
    if v is None and init:
        raise ValueError("v should be provided!")

    #if init is true kickstart the leapfrog algorithm
    if init:
        v = v + dt / 2 * F
        return r, v, None
                                
    r = r + dt * v
    v = v + dt * F 
    vstore = v - dt / 2 * F

    return r, v, vstore
