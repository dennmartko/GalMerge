import numpy as np

v_async = None
rout = None
vout = None

def leapfrog(r, F, v=None, dt=0.001, init=False):
    #v is necessary for initialization after that it's not needed as the
    #function stores v_async in between calls
    if v is None and init:
        raise ValueError("v should be provided!")

    #if init is true kickstart the leapfrog algorithm
    if init:
        global v_async
        v_async = v - dt / 2 * F #compute asynchronized velocities to kickstart the integrator (Euler method is
                                 #used for this)

        #store the initial values of r and v
        global rout
        global vout
        rout = np.array([r])
        vout = np.array([v])

    #update r and v_async using leapfrog
    v_async = v_async + dt * F
    r = r + dt * v_async
    
    v = v_async + dt / 2 * F #resynchronize v for storage

    #store r and v
    rout = np.append(rout, r)
    vout = np.append(vout, v)

    return r, v


"""
class leapfrog:
    def __init__(self, r, v, F, dt=0.001):
        self.r = r
        self.v = v
        self.v_async = self.v - dt/2*F #compute asynchronized velocities to kickstart the integrator

        self.dt  = dt #set timestep

        #storage arrays
        self.rout = np.array([r])
        self.vout = np.array([v])

    def __call__(self, F):
        self.v_async = self.v_async + self.dt*F
        self.r = self.r + self.dt*self.v_async

        #store in output arrays (to do! make a check to see if we have a large enough block for storage to file)
        self.rout = np.append(self.rout, self.r)
        self.vout = np.append(self.vout, self.v_async + self.dt/2*F)

        return self.r
"""

if __name__ == "__main__":
    r = np.arange(1,10,1).reshape(3,3)
    v = np.arange(1,10,1).reshape(3,3)
    F = np.arange(1,10,1).reshape(3,3)

    r, v = leapfrog(r, F, v=v, dt=0.001, init=True)
    print(r, v)
    r, v = leapfrog(r, F, dt=0.001)
    print(r, v)