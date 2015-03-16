#-----------------------------------------------------------------------------
#      
#       Paper: When Crowdsourcing Fails: A Study of Expertise on Crowdsourced 
#              Design Evaluation
#       Author: Alex Burnap - aburnap@umich.edu
#       Date: October 10, 2014
#       License: Apache v2
#       Description: Model definition for creating Bayesian network crowd
#       consensus model
#
#-----------------------------------------------------------------------------

import numpy as np
import pymc
import scipy.stats as stats

def create_model(evaluation_matrix, num_participants, num_designs):
    """
    Function creates Bayesian network model defition as dict for PyMC, called
    by simulation_X.py.

    Input: evaluation matrix, size of crowd, and number of designs

    Output: Dict for PyMC

    Note: Current hyperparameters are hard coded as in paper
    """
    #--------------- Data Manipulation of Evaluation Matrix-------------------
    indices = np.nonzero(evaluation_matrix)
    participant_indices, design_indices = indices[0], indices[1]
    observed_evaluations = evaluation_matrix.reshape(num_participants*num_designs)
    observed_evaluations = np.ma.masked_equal(observed_evaluations,0).compressed()
    observed_evaluations = (observed_evaluations-1)/4.0

    #--- 1st Level --- Hyperparameters of Priors -----------------------------
    ability_mu_prior = 0.5
    ability_tau_prior = 0.1
    logistic_scale_mu = 0.07
    logistic_scale_tau = 1.0
    criteria_score_mu_prior = 0.5
    criteria_score_tau_prior = 0.1

    #--- 2nd Level --- Ability, Difficulty, Logistic Scale, Inv-Wishart Var --
    """
    Currently only each participant has it's own node, there is common node
    for difficulty, logistic scale, and inv_wishart_var
    """
    ability_vector = pymc.TruncatedNormal('ability', mu=ability_mu_prior, 
            tau=ability_tau_prior, a=0, b=1, value=.5*np.ones(num_participants))

    design_difficulty_num = pymc.TruncatedNormal('design_difficulty', 
            mu=0.5, tau=1.0, a=0.3, b=0.7, value=0.5)

    logistic_scale_num = pymc.TruncatedNormal('logistic_scale', mu=logistic_scale_mu, 
            tau=logistic_scale_tau, a=.01, b=.2, value=.07)#, value=.1*np.ones(num_participants))

    inv_gamma_var = .01 # turn this to density later

    #--- 3rd Level ---- Logistic, Alpha, Beta Deterministic ------------------
    @pymc.deterministic
    def logistic_det(ability=ability_vector,  difficulty=design_difficulty_num, scale=logistic_scale_num):
        sigma = np.array(1 - stats.logistic.cdf(ability-difficulty,0,scale)).clip(
                np.spacing(1)*10, 1e6) #this is done to prevent dividing by 0
        return sigma

    @pymc.deterministic
    def alpha_det(E=logistic_det, V=inv_gamma_var):
        return (E**2)/V + 2

    @pymc.deterministic
    def beta_det(E=logistic_det, V=inv_gamma_var):
        return (E*((E**2)/V + 1))

    #--- 4th Level --- Inverse-Gamma and True Score --------------------------
    criteria_score_vector = pymc.TruncatedNormal('criteria_score', mu=criteria_score_mu_prior,
            tau=criteria_score_tau_prior, a=0, b=1, value=.5*np.ones(num_designs))

    inverse_gamma_vector = pymc.InverseGamma('inverse_gamma', alpha=alpha_det, beta=beta_det, 
            value=0.5*np.ones(num_participants))

    #--- 5th Level ---- Evaluations  -------------------------------
    y = pymc.TruncatedNormal('y', mu=criteria_score_vector[design_indices], 
            tau=1/(inverse_gamma_vector[participant_indices]**2), 
            a=0, b=1, value=observed_evaluations, observed=True)

    #--- Return All MCMC Objects ---------------------------------------------
    return {'y':y , 
            'criteria_score_vector': criteria_score_vector,
            'inverse_gamma_vector': inverse_gamma_vector,
            'alpha_det': alpha_det, 
            'beta_det': beta_det, 
            'logistic_det': logistic_det,
            'logistic_scale_num': logistic_scale_num, 
            'ability_vector':ability_vector, 
            'design_difficulty_num':design_difficulty_num}
