#-----------------------------------------------------------------------------
#      
#       Paper: When Crowdsourcing Fails: A Study of Expertise on Crowdsourced 
#              Design Evaluation
#       Author: Alex Burnap - aburnap@umich.edu
#       Date: October 10, 2014
#       License: Apache v2
#       Description: Helper functions used for processing data
#
#-----------------------------------------------------------------------------

from scipy.stats import kendalltau
import numpy as np

def sort(array):
	temp = array.argsort()
	ranks = np.empty(len(array), int)
	ranks[temp] = np.arange(len(array))
	return ranks

def rescale_1_5(array):
	return (array-np.min(array))/(np.max(array)-np.min(array))*4+1

def rescale_0_1(array):
	return (array-np.min(array))/(np.max(array)-np.min(array))

def rescale_int(row):
    row_new = np.zeros(row.shape)
    count = 0
    for i in xrange(5):
        ids = np.nonzero(row==i+1)[0]
        if len(ids)>0:
            row_new[ids] = count
            count += 1
    return row_new

def msqe(x,y):
	return np.average((x-y)**2)

def make_genders(x):
	np.char.strip(gender_vector)=='MA'
	pass

def make_ages(x):
	for i,row in enumerate(x):
		if row=='AA' or row=='A':
			age=16
		elif row=='BB' or row=='B':
			age=20
		elif row=='CC' or row=='C':
			age=23.5
		elif row=='DD' or row=='D':
			age=27.5
		elif row=='EE' or row=='E':
			age=32.5
		elif row=='FF' or row=='F':
			age=40.5
		elif row=='GG' or row=='G':
			age=50.5
		elif row=='HH' or row=='H':
			age=60.5
		elif row=='II' or row=='I':
			age=70.7
		else:
			print "found a bad character"
			raise Exception
		x[i]=age
	return x
			

