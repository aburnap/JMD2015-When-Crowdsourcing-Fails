#-----------------------------------------------------------------------------
#      
#       Paper: When Crowdsourcing Fails: A Study of Expertise on Crowdsourced 
#              Design Evaluation
#       Author: Alex Burnap - aburnap@umich.edu
#       Date: October 10, 2014
#       License: Apache v2
#       Description: Human crowd study data - Bayesian network crowd consensus
#       model benchmarked against averaging.  Although results are stochastic,
#       the results used in the paper are located in ./human_crowd_results/ 
#
#-----------------------------------------------------------------------------
import numpy as np
import pymc
import model
from analysis_functions import *
import csv

#---------------------------- Get Run Settings --------------------------------
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-mcmc', '--mcmc', type=int, default=100000)
parser.add_argument('-burn', '--burn', type=int, default = 50000)
parser.add_argument('-experts', '--num_experts', type=int, default=0)
parser.add_argument('-data', '--dataset', default="./processed_data/new_data_matrix.csv")
parser.add_argument('-rescale',dest='rescale',action='store_true')
parser.add_argument('-no-rescale',dest='rescale',action='store_false')
parser.add_argument('-no-map', dest='use_map', action='store_false')
parser.set_defaults(use_map=True)
parser.set_defaults(rescale=False)        
args = parser.parse_args()

#---------------------------- True Bracket Scores -----------------------------
raw_scores = -np.loadtxt("./processed_data/raw_strengths_vector.csv", delimiter=',')
true_scores = (raw_scores-np.min(raw_scores))/(np.max(raw_scores)-np.min(raw_scores))*4+1

#---------------------------- Loop Setttings ----------------------------------
rescale_bool = True
inject_experts_bool = False
use_map_bool = False
max_experts = 25
mcmc_iter = 5000000
mcmc_burn = 2500000
num_experts=args.num_experts

for i in xrange(2):
    print '---------- Iteration %i -----------------------------' % (i+1)
    #---------------------------- Get Data ----------------------------------------
    evaluations_matrix = np.loadtxt(args.dataset, delimiter=',')
    if rescale_bool:
        evaluations_matrix = np.array([rescale_1_5(row) for row in evaluations_matrix])
    def inject_experts(evaluations_matrix, num_experts, true_scores):
	new_rows = np.array([true_scores for _ in range(num_experts)])
        new_rows = new_rows.clip(1,5)
        return np.concatenate((evaluations_matrix, new_rows))
    if inject_experts_bool:
        if num_experts > 0:
            evaluations_matrix = inject_experts(evaluations_matrix, num_experts, true_scores)
    num_participants, num_designs = np.shape(evaluations_matrix)
    print '---------- Data Loaded ----------------------------------------------'

    #--------------- Averaging Scores -----------------------------------------
    averaging_scores = np.array([np.average(evaluations_matrix[:,i]) for i in xrange(num_designs)])

    #--------------- Bayesian Network Scores and Abilities ------------------------
    raw_model = model.create_model(evaluations_matrix, num_participants, num_designs)
    model_instance = pymc.Model(raw_model)
    print '---------- Bayesian Network Model Setup -----------------------------'

    # Initial Values Set by MAP
    if use_map_bool:
        pymc.MAP(model_instance).fit(method='fmin_powell')
    print '---------- Finished Running MAP to Set MCMC Initial Values ----------'
    # Run MCMC
    print '--------------------------- Starting MCMC ---------------------------'
    M = pymc.MCMC(model_instance)
    M.sample(mcmc_iter, mcmc_burn, thin=5, verbose=0)

    bayesian_network_scores = np.transpose(M.criteria_score_vector.stats()['mean'])*4+1
    bayesian_network_abilities = np.transpose(M.ability_vector.stats()['mean'])

    #print 'Mean Squared Errors'
    averaging_MSqE = np.average((np.array(true_scores) - np.array(averaging_scores))**2)
    bayesian_network_MSqE = np.average((np.array(true_scores) - np.array(bayesian_network_scores))**2)
    #print '--------------------------'
    print 'Averaging is Better' if averaging_MSqE < bayesian_network_MSqE else 'Bayesian Network is Better'

    with open("./human_crowd_results/baseline_BN_vs_Averaging.csv","a") as csvfile:
        results=csv.writer(csvfile)
        results.writerow([num_experts, averaging_MSqE, bayesian_network_MSqE])
