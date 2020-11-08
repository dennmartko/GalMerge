# This module contains all the functions and objects to run the Barnes Hut algorithm.

import numpy as np
import time

import sys
from multiprocessing import Process, Pipe, cpu_count, Array
from ctypes import c_double

#imports from own modules
import constants as const
from BH_utils.OFuncs import Tree_template_init, CM_Handler, NewCellGeom, get_condr, GForce, BHF_handler
from BH_utils.PFuncs import CellPlotter

#Prototype of a cell object
class Cell(object):
	__slots__ = ('midR', 'L', 'parent', 'M', 'R_CM', 'daughters')
	def __init__(self, midR, L, parent=None, M=None, R_CM=None):
		self.parent = parent #parent of the current cell
		self.daughters = None #daughters of the current cell
		
		#Physical quantities
		self.M = M #total mass contained within the cell
		self.R_CM = R_CM #location of the center of mass of the cell

		#Geometrical quantities
		self.midR = midR #coordinate location of the cell's center
		self.L = L #length of the cell's sides


#Prototype of a particle object
class Particle(object):
	__slots__ = ('r', 'v', 'm')
	def __init__(self, r, v, m=None):
		# Position, velocity and mass
		self.r = r
		self.v = v

		if m is None:
			self.m = 1 #give the particle the mass of the Sun if m
					   #is not provided
		else:
			self.m = m


def rmParticles(rdd1, rdd2, rdd3, rdd4, rdd5, rdd6, rdd7, rdd8, particles1, particles2, particles3, particles4, particles5, particles6, particles7, particles8):
	'''Effective elimination of stars (in rdd{i}) that have been detected to not be in a given cell {i}
		As a result the stars in particles{i} that are inside cell{i} gets reduced in size / is updated
	'''
	# np.delete() does not work with empty lists
	if len(rdd1) != 0:
		particles1 = np.delete(particles1, rdd1, axis=0)

	if len(rdd2) != 0:
		particles2 = np.delete(particles2, rdd2, axis=0)

	if len(rdd3) != 0:
		particles3 = np.delete(particles3, rdd3, axis=0)

	if len(rdd4) != 0:
		particles4 = np.delete(particles4, rdd4, axis=0)

	if len(rdd5) != 0:
		particles5 = np.delete(particles5, rdd5, axis=0)

	if len(rdd6) != 0:
		particles6 = np.delete(particles6, rdd6, axis=0)

	if len(rdd7) != 0:
		particles7 = np.delete(particles7, rdd7, axis=0)

	if len(rdd8) != 0:
		particles8 = np.delete(particles8, rdd8, axis=0)

	return particles1, particles2, particles3, particles4, particles5, particles6, particles7, particles8 


def Tree(node, particles, obj=None):
	'''Tree generator of the Barnes-Hut Algorithm.
		Cells are split to 2^d cells, with d=dimension, if more than 1 star has been detected within them.
	'''
	if obj is not None:
		obj.append(node) # append the created node to the object list

	# Hard copy the particle array
	particles1 = particles.copy()
	particles2 = particles.copy()
	particles3 = particles.copy()
	particles4 = particles.copy()
	particles5 = particles.copy()
	particles6 = particles.copy()
	particles7 = particles.copy()
	particles8 = particles.copy()

	# Redundant particles for each quadrant (the number in the variable name
	# refers to the ith quadrant)
	rdd1 = []
	rdd1app = rdd1.append
	rdd2 = []
	rdd2app = rdd2.append
	rdd3 = []
	rdd3app = rdd3.append
	rdd4 = []
	rdd4app = rdd4.append
	rdd5 = []
	rdd5app = rdd5.append
	rdd6 = []
	rdd6app = rdd6.append
	rdd7 = []
	rdd7app = rdd7.append
	rdd8 = []
	rdd8app = rdd8.append

	num1, num2, num3, num4, num5, num6, num7, num8, M1, M2, M3, M4, M5, M6, M7, M8 = Tree_template_init()

	node.daughters = []

	# Initialize particle counter
	pcount = 0

	# Check if more than 1 particles inside square
	for indx, p in enumerate(particles):
		r = p.r
		m = p.m
		condr = get_condr(r, node.L, node.midR) #condition r
		#2D slice above z=0
		if 1 > condr[0] > 0 and 1 > condr[1] > 0 and 1 > condr[2] > 0:
			pcount += 1
			rdd2app(indx)
			rdd3app(indx)
			rdd4app(indx)
			rdd5app(indx)
			rdd6app(indx)
			rdd7app(indx)
			rdd8app(indx)

			num1, M1 = CM_Handler(num1,r,m,M1)	# compute position center of mass nummerator and denominator. 
												# Denominator is equal to the total mass M{i} of the ith quadrant or octant.
		elif -1 < condr[0] < 0 and 1 > condr[1] > 0 and 1 > condr[2] > 0:
			pcount += 1
			rdd1app(indx)
			rdd3app(indx)
			rdd4app(indx)
			rdd5app(indx)
			rdd6app(indx)
			rdd7app(indx)
			rdd8app(indx)

			num2, M2 = CM_Handler(num2,r,m,M2)
		elif -1 < condr[0] < 0 and -1 < condr[1] < 0 and 1 > condr[2] > 0:
			pcount += 1
			rdd1app(indx)
			rdd2app(indx)
			rdd4app(indx)
			rdd5app(indx)
			rdd6app(indx)
			rdd7app(indx)
			rdd8app(indx)

			num3, M3 = CM_Handler(num3,r,m,M3)
		elif 1 > condr[0] > 0 and -1 < condr[1] < 0 and 1 > condr[2] > 0:
			pcount += 1
			rdd1app(indx)
			rdd2app(indx)
			rdd3app(indx)
			rdd5app(indx)
			rdd6app(indx)
			rdd7app(indx)
			rdd8app(indx)

			num4, M4 = CM_Handler(num4,r,m,M4)
		#2D slice beneath z=0
		elif 1 > condr[0] > 0 and 1 > condr[1] > 0 and -1 < condr[2] < 0:
			pcount += 1
			rdd1app(indx)
			rdd2app(indx)
			rdd3app(indx)
			rdd4app(indx)
			rdd6app(indx)
			rdd7app(indx)
			rdd8app(indx)

			num5, M5 = CM_Handler(num5,r,m,M5)
		elif -1 < condr[0] < 0 and 1 > condr[1] > 0 and -1 < condr[2] < 0:
			pcount += 1
			rdd1app(indx)
			rdd2app(indx)
			rdd3app(indx)
			rdd4app(indx)
			rdd5app(indx)
			rdd7app(indx)
			rdd8app(indx)

			num6, M6 = CM_Handler(num6,r,m,M6)
		elif -1 < condr[0] < 0 and -1 < condr[1] < 0 and -1 < condr[2] < 0:
			pcount += 1
			rdd1app(indx)
			rdd2app(indx)
			rdd3app(indx)
			rdd4app(indx)
			rdd5app(indx)
			rdd6app(indx)
			rdd8app(indx)

			num7, M7 = CM_Handler(num7,r,m,M7)
		elif 1 > condr[0] > 0 and -1 < condr[1] < 0 and -1 < condr[2] < 0:
			pcount += 1
			rdd1app(indx)
			rdd2app(indx)
			rdd3app(indx)
			rdd4app(indx)
			rdd5app(indx)
			rdd6app(indx)
			rdd7app(indx)

			num8, M8 = CM_Handler(num8,r,m,M8)

	# If theres more than one particle in a node, we can create new nodes!
	if pcount > 1:
		#remove redundant particles from particles arrays
		particles1, particles2, particles3, particles4, particles5, particles6, particles7, particles8 = rmParticles(np.array(rdd1), np.array(rdd2), np.array(rdd3), np.array(rdd4), np.array(rdd5),
										np.array(rdd6),np.array(rdd7), np.array(rdd8), particles1, particles2, particles3, particles4, particles5, particles6, particles7, particles8)

		# if a potential cell's mass is nonzero create it!
		# This effectively prevents empty cells being created.
		if M1 != 0:
			newmidR, newL = NewCellGeom(node.midR, node.L, 1)
			D1 = Cell(newmidR, newL, parent=node, M = M1, R_CM = num1 / M1)
			node.daughters.append(D1)
			Tree(D1, particles1, obj)
		if M2 != 0:
			newmidR, newL = NewCellGeom(node.midR, node.L, 2)
			D2 = Cell(newmidR, newL, parent=node, M = M2, R_CM = num2 / M2)
			node.daughters.append(D2)
			Tree(D2, particles2, obj)
		if M3 != 0:
			newmidR, newL = NewCellGeom(node.midR, node.L, 3)
			D3 = Cell(newmidR, newL, parent=node, M = M3, R_CM = num3 / M3)
			node.daughters.append(D3)
			Tree(D3, particles3, obj)
		if M4 != 0:
			newmidR, newL = NewCellGeom(node.midR, node.L, 4)
			D4 = Cell(newmidR, newL, parent=node, M = M4, R_CM = num4 / M4)
			node.daughters.append(D4)
			Tree(D4, particles4, obj)
		if M5 != 0:
			newmidR, newL = NewCellGeom(node.midR, node.L, 5)
			D5 = Cell(newmidR, newL, parent=node, M = M5, R_CM = num5 / M5)
			node.daughters.append(D5)
			Tree(D5, particles5, obj)
		if M6 != 0:
			newmidR, newL = NewCellGeom(node.midR, node.L, 6)
			D6 = Cell(newmidR, newL, parent=node, M = M6, R_CM = num6 / M6)
			node.daughters.append(D6)
			Tree(D6, particles6, obj)
		if M7 != 0:
			newmidR, newL = NewCellGeom(node.midR, node.L, 7)
			D7 = Cell(newmidR, newL, parent=node, M = M7, R_CM = num7 / M7)
			node.daughters.append(D7)
			Tree(D7, particles7, obj)
		if M8 != 0:
			newmidR, newL = NewCellGeom(node.midR, node.L, 8)
			D8 = Cell(newmidR, newL, parent=node, M = M8, R_CM = num8 / M8)
			node.daughters.append(D8)
			Tree(D8, particles8, obj)


# Functions for computing the gravitational force on a single particle
def BHF(node, rp, force_arr, θ, ε, SMBHS):
	daughters = node.daughters
	if not (node.R_CM == rp).all():
		if BHF_handler(rp, node.R_CM, node.L, θ) or daughters == []:
			force_arr.append(GForce(node.M, rp, node.R_CM, SMBHS, ε=ε))
		else:
			for i in range(len(daughters)):
				BHF(daughters[i], rp, force_arr, θ, ε=ε, SMBHS=SMBHS)

def BHF_kickstart(ROOT, particles, Forces=None, θ=0.5, ε=0.1, SMBHS=None, conn=None):
	#Forces will be None if the platform is 'win32'.  In that case we should
	#receive Forces through a duplex Pipe.
	if Forces is None and conn is not None:
		Forces = conn.recv() #waits until there's something to receive

	#iterate through all particles
	for i, p in enumerate(particles):
		force_arr = []
		BHF(ROOT, p.r, force_arr, θ, ε=ε, SMBHS=SMBHS)
		Fg = np.sum(np.array(force_arr), axis=0)
		Forces[i,:] = Fg.astype(dtype=c_double)

	#send the updated Forces array back through the Pipe
	if conn is not None:
		conn.send(Forces)
		conn.close()

#receive something via a connection and close the connection
def connection_receiveAndClose(conn):
	tmp = conn.recv()
	conn.close()
	return tmp

#function to join and terminate the processes
def processes_joinAndTerminate(processes):
	#join processes
	for p in processes:
		p.join()

	#terminate processes
	for p in processes:
		p.terminate()