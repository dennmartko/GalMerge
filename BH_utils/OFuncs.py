import numpy as np
from numba import jit,njit
import time
#imports from own modules
import constants as const

''' OPTIMIZED FUNCTIONS FOR BH TREE COMPUTATIONS '''

@njit
def Tree_template_init():
	'''Function that serves templates to kickstart each Tree recursion'''
	# Temporary memory where we store numerator of R_CM
	num1, num2, num3, num4, num5, num6, num7, num8 = np.zeros((8,3), dtype=np.float64)

	# Total mass of each cell
	M1 = M2 = M3 = M4 = M5 = M6 = M7 = M8 = 0

	return num1, num2, num3, num4, num5, num6, num7, num8, M1, M2, M3, M4, M5, M6, M7, M8 

@njit
def CM_Handler(num,r,m,M):
	'''Compute the numerator and denominator seperately during iteration process '''
	return (num + m * r, M + m)

@njit
def NewCellGeom(midR,L,order):
	'''Compute the center coordinates of a splitted cell'''
	#2D slice above z=0
	if order == 1:
		newmidR = midR + np.array([L / 4, L / 4, L / 4])
	if order == 2:
		newmidR = midR + np.array([-L / 4, L / 4, L / 4])
	if order == 3:
		newmidR = midR + np.array([-L / 4, -L / 4, L / 4])
	if order == 4:
		newmidR = midR + np.array([L / 4, -L / 4, L / 4])

	#2D slice beneath z=0
	if order == 5:
		newmidR = midR + np.array([L / 4, L / 4, -L / 4])
	if order == 6:
		newmidR = midR + np.array([-L / 4, L / 4, -L / 4])
	if order == 7:
		newmidR = midR + np.array([-L / 4, -L / 4, -L / 4])
	if order == 8:
		newmidR = midR + np.array([L / 4, -L / 4, -L / 4])	
	newL = L / 2
	return newmidR, newL

@njit
def get_condr(r, L, midR):
	'''Transform position of particle into a simple 1D conditional array which can effectively be used'''
	return 2 * (r - midR) / L


def GForce(M, rp, Rcm, SMBHS, ε=0.1):
	'''Force calculation for each particle'''
	r = rp-Rcm
	Fg = -1 * (const.G_ * M) * r / (np.linalg.norm(r) ** 2 + ε ** 2) ** (3 / 2) # ε is softening parameter defaulting to 0.1

	#compute gravitational force of the supermassive black holes
	if SMBHS is not None:
		for SMBH in SMBHS:
			rSMBH = rp - SMBH.r #vector pointing radially away from the supermassive blackhole
			Fg -= (const.G_ * SMBH.m) * rSMBH / (np.linalg.norm(rSMBH) ** 2 + SMBH.ε ** 2) ** (3/2)
			#Fg -= (const.G_ * SMBH.m) * rSMBH / (np.linalg.norm(rSMBH)*(np.linalg.norm(rSMBH) + r0)**2)

	return Fg

@njit(fastmath=True)
def BHF_handler(rp, Rcm, L, θ):
	'''Calculation of the accuracy parameter θ given cell&particle parameters'''
	r = rp - Rcm

	D = (r[0] ** 2 + r[1] ** 2 + r[2] ** 2) ** (1 / 2)


	if D == 0:
		return False
	elif L / D <= θ:
		return True
	else:
		return False
