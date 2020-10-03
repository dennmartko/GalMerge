import numpy as np
from numba import jit,njit

#imports from own modules
import constants as const

''' OPTIMIZED FUNCTIONS FOR TREE COMPUTATIONS '''

@njit
def Tree_template_init():
	'''Function that serves templates to kickstart each Tree recursion'''
	# Temporary memory where we store numerator of R_CM
	num1, num2, num3, num4 = np.zeros((4,2), dtype=np.float64)

	# Total mass of each cell
	M1 = M2 = M3 = M4 = 0

	return num1, num2, num3, num4, M1, M2, M3, M4

@njit
def CM_Handler(num,r,m,M):
	'''Compute the numerator and denominator seperately during iteration process '''
	return (num + m * r, M + m)

@njit
def NewCellGeom(midR,L,order):
	'''Compute the center coordinates of a splitted cell'''
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

@njit
def get_condr(r, L, midR):
	'''Transform position of particle into a simple 1D conditional array which can effectively be used'''
	return 2 * (r - midR) / L


@njit(fastmath=True)
def GForce(M, rp, Rcm):
	r = rp - Rcm
	Fg = -1*(const.G_ * M) / (r[0] ** 2 + r[1] ** 2) ** (3 / 2) * (r)
	return Fg

@njit(fastmath=True)
def BHF_handler(rp, Rcm, L, θ):
	r = rp - Rcm
	D = (r[0] ** 2 + r[1] ** 2) ** (1 / 2)

	if D == 0:
		return False
	elif L / D <= θ:
		return True
	else:
		return False
