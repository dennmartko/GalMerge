import numpy as np
import numba

cimport numpy as np
cimport cython

#imports from own modules
import constants as const

''' OPTIMIZED FUNCTIONS FOR TREE COMPUTATIONS '''
def Tree_template_init():
	# Temporary memory where we store numerator of R_CM
	cdef np.ndarray num1,num2,num3,num4
	num1, num2, num3, num4 = np.zeros((4,2), dtype=np.float64)

	# Total mass of each cell
	cdef double M1 = 0.0
	cdef double M2 = 0.0
	cdef double M3 = 0.0
	cdef double M4 = 0.0

	return num1, num2, num3, num4, M1, M2, M3, M4

@cython.boundscheck(False)
@cython.wraparound(False)
def CM_Handler(np.ndarray num,np.ndarray r,float m,float M):
	return (num + m*r, M + m)

@cython.boundscheck(False)
@cython.wraparound(False)
def NewCellGeom(np.ndarray midR,float L,int order):
	cdef np.ndarray newmidR
	cdef float newL

	if order == 1:
		newmidR = midR + np.array([L / 4, L / 4])
	if order == 2:
		newmidR = midR + np.array([-L / 4, L / 4])
	if order == 3:
		newmidR = midR + np.array([-L / 4, -L / 4])
	if order == 4:
		newmidR = midR + np.array([L / 4, -L / 4])
	
	newL = L / 2
	return newmidR, newL

@cython.boundscheck(False)
@cython.wraparound(False)
def get_condr(np.ndarray r, float L,np.ndarray midR):
	return 2*(r-midR)/L

def rmParticles(np.ndarray rdd1,np.ndarray rdd2,np.ndarray rdd3,np.ndarray rdd4,np.ndarray particles1,np.ndarray particles2,np.ndarray  particles3,np.ndarray  particles4):
	# np.delete() does not work with empty lists

	if len(rdd1) != 0:
		particles1 = np.delete(particles1, rdd1, axis=0)

	if len(rdd2) != 0:
		particles2 = np.delete(particles2, rdd2, axis=0)

	if len(rdd3) != 0:
		particles3 = np.delete(particles3, rdd3, axis=0)

	if len(rdd4) != 0:
		particles4 = np.delete(particles4, rdd4, axis=0)

	return particles1, particles2, particles3, particles4
