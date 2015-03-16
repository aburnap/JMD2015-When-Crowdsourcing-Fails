#-----------------------------------------------------------------------------
#      
#       Paper: When Crowdsourcing Fails: A Study of Expertise on Crowdsourced 
#              Design Evaluation
#       Author: Alex Burnap - aburnap@umich.edu
#       Date: October 10, 2014
#       License: Apache v2
#       Description: Class definitions for crowdsourced evaluation process
#
#-----------------------------------------------------------------------------

import numpy as np
import scipy.stats as stats
from sklearn.mixture import GMM

#---------------------------------------------------------------------------------------------  
class Participant(object):
    """
    Each participant has attributes ability and cluster_id which determines bias
    """
    def __init__(self, true_ability=None, cluster_id=-1):
        self.true_ability = true_ability
        self.cluster_id = cluster_id

#---------------------------------------------------------------------------------------------
class Cluster(object):
    """
    Each participant belongs to a cluster, which determines the mean offset of evaluations
    for a specific design.
    """
    def __init__(self, biases):
        self.biases = biases # num_designs x 1

#--------------------------------------------------------------------------------------------- 
class Design(object):
    """
    Each Design has attributes- true_criteria_score , true_evaluation_difficulty
    """
    def __init__(self,true_criteria_score, true_evaluation_difficulty):
        self.true_criteria_score = true_criteria_score
        self.true_evaluation_difficulty= true_evaluation_difficulty

#--------------------------------------------------------------------------------------------- 
class Environment(object):
    """
    This Environment contains participants, designs, and evaluations.
    Takes arguements num_participants, num_designs, crowd_composition
    """
    def __init__(self, cp, dp, ep, clp):
        self.cp = cp
        self.dp = dp
        self.ep = ep
        self.clp = clp
        self.num_participants = cp['num_participants']
        self.num_designs = dp['num_designs']
        self.num_subcriteria = dp['num_subcriteria']
        self.num_clusters = clp['num_clusters']
        self.participants = self.create_participants()
        self.designs = self.create_designs()
        self.clusters = self.create_clusters()
        self.evaluations_matrix = np.zeros([cp['num_participants'], dp['num_designs']])
        self.temp_3D_array = np.zeros([self.ep['num_queries_per_participant'], self.num_participants, self.num_designs])

    def create_participants(self):
        ability_vector = self.create_true_participant_abilities()
        cluster_vector = self.create_cluster_vector()
        return [Participant(true_ability=ability_vector[i], cluster_id=cluster_vector[i]) for i in xrange(self.num_participants)]

    def create_designs(self):
        criteria_score_vector = self.create_true_design_criteria_scores()
        evaluation_difficulty_vector = self.create_true_design_evaluation_difficulties()
        return [Design(criteria_score_vector[i], evaluation_difficulty_vector[i]) for i in xrange(self.num_designs)]

    def create_clusters(self):
        return [Cluster(self.clp['cluster_biases'][i]) for i in xrange(self.num_clusters)]
    #---------------------------------------------------------------------------------------------     
    def create_true_participant_abilities(self):
        """
        This function returns a vector of (currently scalar) abilities given 1 of 3 types of crowd makeup.
        Arguements are a string crowd_makeup = 'mixture' or 'homogenous' or 'random'
        If 'homogeneous', then a 2-tuple of the mean ability and its variance.
        If 'random', no additional tuple is required.
        If 'mixture', then a 6-tuple of the low and high ability value means, their variances, their mixing coefficients is required
        """
        if self.cp['crowd_makeup'] == 'homogeneous':
#            return np.random.normal(self.cp['homo_mean'], self.cp['homo_std_dev'], self.cp['num_participants']).clip(0,1)
            return stats.truncnorm(-self.cp['homo_mean']/self.cp['homo_std_dev'], (1-self.cp['homo_mean'])/self.cp['homo_std_dev'], self.cp['homo_mean'], self.cp['homo_std_dev']).rvs(size=self.cp['num_participants'])
        elif self.cp['crowd_makeup'] == 'random':
            return np.random.uniform(0,1,self.cp['num_participants'])
        elif self.cp['crowd_makeup'] == 'mixture':
            gmm = GMM(2, n_iter=1)
            gmm.means_ = np.array([ [self.cp['mixture_means'][0]], [self.cp['mixture_means'][1]]])
            gmm.covars_ = np.array([ [self.cp['mixture_std_dev'][0]], [self.cp['mixture_std_dev'][1]] ]) ** 2
            gmm.weights_ = np.array([self.cp['mixture_coefficients'][0], self.cp['mixture_coefficients'][1]])
            packed = gmm.sample(self.cp['num_participants']).clip(0,1)
            return [packed[i][0] for i in xrange(len(packed))]

    #--------------------------------------------------------------------------------------------- 
    def create_true_design_criteria_scores(self):
        true_criteria_scores=np.zeros(self.num_designs)
        if self.dp['true_design_criteria_score_makeup'] == 'random':
            for i in xrange(self.num_designs):
                true_criteria_scores[i] = np.random.random()
        return true_criteria_scores

    def create_true_design_evaluation_difficulties(self):
        """
        This functions returns a vector of (currently scalar) design evaluation difficulties
        """
        true_evaluation_difficulties=np.zeros(self.num_designs)
        if self.dp['true_design_evaluation_difficulty_makeup'] == 'same':
            for i in xrange(self.num_designs):
                true_evaluation_difficulties[i] = self.dp['true_design_evaluation_difficulty_score']
        return true_evaluation_difficulties

    #---------------------------------------------------------------------------------------------
    def create_cluster_vector(self):
        self.num_clusters # THIS IS ALWAYS 2 FOR NOW
        rand_vec = np.random.random(size=self.num_participants)
        cluster_vector = [0 if elm<self.clp['cluster_proportions'][0] else 1 for elm in rand_vec]
        return cluster_vector

    #---------------------------------------------------------------------------------------------
    def run_evaluations(self):
        """
        This function runs the number of queries per participant on each participant in entire crowd.
        Each query has num_designs_per_query shown to the participant.
        We do not model the effect of information overload with too many designs, or too few either.
        It is an incredibly inefficient function right now, optimize in future if it slows things down too much.
        """
        temp_3D_array = np.zeros([self.ep['num_queries_per_participant'],self.num_participants, self.num_designs])
        for p_ind, participant in enumerate(self.participants):
            for q_ind in xrange(self.ep['num_queries_per_participant']):
                for d_ind in self.random_indices():
                    temp_3D_array[q_ind,p_ind,d_ind] = self.evaluate(participant, self.designs[d_ind], d_ind)
        for i in xrange(self.num_participants):
            for j in xrange(self.num_designs):
                self.evaluations_matrix[i,j] = temp_3D_array[:,i,j].sum()/max(1,len(*np.nonzero(temp_3D_array[:,i,j])))
        self.temp_3D_array = temp_3D_array

        """
        [a[:,i].sum() for i in range(10)]
        for i in range(2):
            for j in range(3):
                print a[:,i,j].sum()
        """

    def evaluate(self, participant, design, design_id):
        """
        Function returns a single evaluation given a participant and a design
        """
        t = design.true_criteria_score
        a = participant.true_ability
        d = design.true_evaluation_difficulty
        b = self.ep['interface_difficulty']
        s = self.ep['logistic_scale']
        error_sigma = 1.0 - stats.logistic.cdf(a-d, b, s)
        evaluation_0_1_unbiased = stats.truncnorm(-t/error_sigma, (1-t)/error_sigma, t, error_sigma).rvs(size=1)[0]
        bias = self.clusters[participant.cluster_id].biases[design_id]
        evaluation_0_1 = np.clip(evaluation_0_1_unbiased + bias, 0, 1)
        evaluation_1_5 = evaluation_0_1 * 4 + 1
        return evaluation_1_5


    def random_indices(self):
        """
        Returns a vector of random indices without replacement. 
        """
        random_indices = []
        while len(random_indices) < self.ep['num_designs_per_query']:
            random_index = np.random.random_integers(0, self.num_designs - 1)
            if random_index not in random_indices:
                random_indices.append(random_index)
        return random_indices
