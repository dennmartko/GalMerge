import numpy as np
from tqdm import tqdm, tqdm_gui
from numba import jit,njit

@njit
def leapfrog(r, F, v, dt):
    #v is necessary for initialization after that it's not needed as the
    #function stores v_async in between calls
    if v is None:
        raise ValueError("v should be provided!")
    
    #this leapfrog assumes v is out of sync
    v = v + dt * F
    r = r + dt * v
    return r, v