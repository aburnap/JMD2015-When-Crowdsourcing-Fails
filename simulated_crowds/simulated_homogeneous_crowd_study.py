#-----------------------------------------------------------------------------
#      
#       Paper: When Crowdsourcing Fails: A Study of Expertise on Crowdsourced 
#              Design Evaluation
#       Author: Alex Burnap - aburnap@umich.edu
#       Date: October 10, 2014
#       License: Apache v2
#       Description: Simulated Crowd Study for Homogeneous Crowds. Used to
#       generate data for Figure 4.
#
#-----------------------------------------------------------------------------
import simulation, model
import numpy as np
import pymc
import csv

#---------------------------------------------------------------------------------------------
# Simulation Global Variables
crowd_parameters = {
        'num_participants' : 60,
        'crowd_makeup' : 'homogeneous',
        'homo_mean' : .3,
        'homo_std_dev' : .1,
        'mixture_means' : (.2, .8),
        'mixture_std_dev' : (.1, .1),
        'mixture_coefficients' : (.9, .1),
        }

design_parameters = {
        'num_designs' : 8,
        'num_subcriteria' : 1,
        'true_design_criteria_score_makeup' : 'random',
        'true_design_evaluation_difficulty_makeup' : 'same',
        'true_design_evaluation_difficulty_score' : .5,
        }

# interface difficulty is the intrinsic bias to all evaluation
# evaluation scale is set where it is for good reason, it is the discrimination factor or slope
evaluation_parameters = {
        'num_queries_per_participant' : 20,
        'num_designs_per_query' : 3,
        'interface_difficulty' : 0,
        'logistic_scale' : .1,
        }
for i in xrange(250):
    print '----------------------------------------------------'
    print "Iteration %i" % (i+1)
    print
    i_homo_mean = np.random.random()
    crowd_parameters['homo_mean']=i_homo_mean
    env=simulation.Environment(crowd_parameters,design_parameters,evaluation_parameters)
    env.run_evaluations()
    raw_model = model.create_model(env.evaluations_matrix, crowd_parameters['num_participants'], design_parameters['num_designs'])
    model_instance = pymc.Model(raw_model)
    # Initial Values Set by MAP
    pymc.MAP(model_instance).fit(method='fmin_powell')
    print '---------- Finished Running MAP to Set MCMC Initial Values ----------'
    # Run MCMC
    print '--------------------------- Starting MCMC ---------------------------'
    M = pymc.MCMC(model_instance)
    M.sample(200000,100000, thin=5, verbose=0)

    true_abilities = [env.participants[i].true_ability for i in xrange(crowd_parameters['num_participants'])]
    true_scores=[(env.designs[i].true_criteria_score*4+1) for i in xrange(design_parameters['num_designs'])]
    estimated_scores = np.transpose(M.criteria_score_vector.stats()['mean'])*4+1
    estimated_abilities = np.transpose(M.ability_vector.stats()['mean'])
    majority_vote_scores = [np.average(env.evaluations_matrix[:,i]) for i in xrange(design_parameters['num_designs'])]
    majority_vote_MSqE = np.average((np.array(true_scores) - np.array(majority_vote_scores))**2)
    estimated_MSqE = np.average((np.array(true_scores) - np.array(estimated_scores))**2)
    estimated_abilities_MSqE = np.average((np.array(true_abilities) - np.array(estimated_abilities))**2)
    estimated_logistic_scale = M.logistic_scale_num.stats()['mean']
    estimated_design_difficulty = M.design_difficulty_num.stats()['mean']

    with open("../simulated_crowd_results/results_homo_mean.csv","a") as csvfile:
            results=csv.writer(csvfile)
            results.writerow([i_homo_mean, majority_vote_MSqE, estimated_MSqE, estimated_abilities_MSqE, estimated_logistic_scale, estimated_design_difficulty])
