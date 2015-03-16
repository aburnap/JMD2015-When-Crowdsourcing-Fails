#-----------------------------------------------------------------------------
#      
#       Paper: When Crowdsourcing Fails: A Study of Expertise on Crowdsourced 
#              Design Evaluation
#       Author: Alex Burnap - aburnap@umich.edu
#       Date: October 10, 2014
#       License: Apache v2
#       Description: Simulated Crowd Study for Heterogeneous Crowds. Used to
#       generate data for Figure 5.
#
#-----------------------------------------------------------------------------

import simulation_heterogeneous as sim
import model
import numpy as np
import pymc
import csv

#-----------------------------------------------------------------------------
# Simulated Crowd Variables
crowd_parameters = {
        'num_participants' : 60,
        'crowd_makeup' : 'homogeneous',
        'homo_mean' : .8,
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

cluster_biases0 = np.zeros(design_parameters['num_designs'])
cluster_biases1 = np.zeros(design_parameters['num_designs'])
cluster_biases1[6] = 0.5

cluster_parameters = {
        'num_clusters' : 2,
        'cluster_proportions' : (.8,.2),
        'cluster_biases' : (cluster_biases0 ,cluster_biases1), # this is on 0_1 scale
        }

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
    i_cluster0_proportion = np.random.random()
    cluster_parameters['cluster_proportions']= (i_cluster0_proportion, 1-i_cluster0_proportion)
    env=sim.Environment(crowd_parameters, design_parameters, evaluation_parameters, cluster_parameters)
    env.designs[6].true_criteria_score = 0.2
    env.run_evaluations()
    raw_model = model.create_model(env.evaluations_matrix, 
            crowd_parameters['num_participants'], 
            design_parameters['num_designs'])
    model_instance = pymc.Model(raw_model)
    # Initial Values Set by MAP
    #pymc.MAP(model_instance).fit(method='fmin_powell')
    print '---------- Finished Running MAP to Set MCMC Initial Values ----------'
    # Run MCMC
    print '--------------------------- Starting MCMC ---------------------------'
    M = pymc.MCMC(model_instance)
    M.sample(200000,100000, thin=5, verbose=0)

    true_abilities = [env.participants[i].true_ability for i in xrange(crowd_parameters['num_participants'])]
    true_scores=[(env.designs[i].true_criteria_score*4+1) for i in xrange(design_parameters['num_designs'])]
    bayesian_network_scores = np.transpose(M.criteria_score_vector.stats()['mean'])*4+1
    bayesian_network_abilities = np.transpose(M.ability_vector.stats()['mean'])
    averaging_scores = [np.average(env.evaluations_matrix[:,i]) for i in xrange(design_parameters['num_designs'])]
    averaging_MSqE = np.average((np.array(true_scores) - np.array(averaging_scores))**2)
    bayesian_network_MSqE = np.average((np.array(true_scores) - np.array(bayesian_network_scores))**2)
    bayesian_network_abilities_MSqE = np.average((np.array(true_abilities) - np.array(bayesian_network_abilities))**2)
    bayesian_network_logistic_scale = M.logistic_scale_num.stats()['mean']
    bayesian_network_design_difficulty = M.design_difficulty_num.stats()['mean']

    with open("./simulated_crowd_results/results_heterogeneous_clusters.csv","a") as csvfile:
            results=csv.writer(csvfile)
            results.writerow([i_cluster0_proportion, averaging_MSqE, 
                bayesian_network_MSqE, 
                bayesian_network_abilities_MSqE, 
                bayesian_network_logistic_scale, 
                bayesian_network_design_difficulty])
