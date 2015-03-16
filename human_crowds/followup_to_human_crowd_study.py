#-----------------------------------------------------------------------------
#      
#       Paper: When Crowdsourcing Fails: A Study of Expertise on Crowdsourced 
#              Design Evaluation
#       Author: Alex Burnap - aburnap@umich.edu
#       Date: October 10, 2014
#       License: Apache v2
#       Description: Cluster the crowd based on evaluations.  Find the cluster
#       errors, including the "expert" cluster. Outputs Fig. 7 and data
#       for Figure 8 and 9. Also prints cluster errors in Table 2.
#
#-----------------------------------------------------------------------------
import numpy as np
import pymc
import scipy
from analysis_functions import *
from sklearn.cluster import DBSCAN
from sklearn.manifold import MDS


#---------------------------- Get Run Settings --------------------------------
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-data', '--dataset', default="./processed_data/new_data_matrix.csv")
parser.add_argument('-rescale',dest='rescale',action='store_true')
parser.add_argument('-no-rescale',dest='rescale',action='store_false')
parser.set_defaults(rescale=True)        
args = parser.parse_args()

#---------------------------- True Bracket Scores -----------------------------
raw_scores = -np.loadtxt("./processed_data/raw_strengths_vector.csv", delimiter=',')
true_scores = (raw_scores-np.min(raw_scores))/(np.max(raw_scores)-np.min(raw_scores))*4+1

#---------------------------- Get Data ----------------------------------------
data_matrix = np.loadtxt(args.dataset, delimiter=',')

if args.rescale:
    data_matrix = np.array([rescale_int(row) for row in data_matrix])

num_participants, num_designs = np.shape(data_matrix)
print '---------- Data Loaded ----------------------------------------------'

exp = []
evaluations_matrix = np.delete(data_matrix,exp,1)

# true high ability people
true_similarity = np.zeros((num_participants))
for i in range(num_participants):
    try:
        true_similarity[i] = scipy.stats.stats.kendalltau(evaluations_matrix[i,:],
                np.delete(true_scores,exp,0))[0]
    except FloatingPointError as e:
        print str(e) + "with index " + str(i)
good_ppl_id = np.nonzero(true_similarity>0.75)[0]

#--------------- Clustering based on evaluations -----------------------------------------
similarity = np.ones((num_participants,num_participants))
for i in range(num_participants-1):
    for j in range(i+1,num_participants):
            similarity[i,j] = scipy.stats.stats.kendalltau(evaluations_matrix[i,:],
                    evaluations_matrix[j,:])[0]
            similarity[j,i] = similarity[i,j]
db = DBSCAN(eps=1.4, min_samples=4).fit(similarity)
core_samples = db.core_sample_indices_
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

# Plot result
import pylab as pl
from itertools import cycle

pl.close('all')
pl.figure(1)
pl.clf()

mds = MDS(n_components=2, metric=True, dissimilarity='precomputed')
X = mds.fit(1-similarity).embedding_
# Black removed and is used for noise instead.
colors = cycle('bgrcmybgrcmybgrcmybgrcmy')
for k, col in zip(set(labels), colors):
    if k == -1:
        # Black used for noise.
        col = 'k'
        markersize = 6
    class_members = [index[0] for index in np.argwhere(labels == k)]
    cluster_core_samples = [index for index in core_samples
                            if labels[index] == k]
    markersize = 6
    for index in class_members:
        x = X[index]
        #if index in core_samples and k != -1:
        if k != -1:
            markersize = 14
        markeredgewidth=1
        markeredgecolor='k' 
        
        pl.plot(x[0], x[1], 'o', markerfacecolor=col,
                markeredgecolor=markeredgecolor, 
                markersize=markersize, markeredgewidth=markeredgewidth)

pl.title('Estimated number of clusters: %d' % n_clusters_)

#--------------- Errors of the Clusters  --------------
data_matrix = np.loadtxt("./processed_data/new_data_matrix.csv", delimiter=',')
#data_matrix = np.delete(data_matrix, 2, 1)
data_matrix = np.array([rescale_1_5(row) for row in data_matrix])
colors = cycle('bgrcmybgrcmybgrcmybgrcmy')

for label_id in xrange(0,5):
    print "Cluster error #%s" % label_id
    print np.average((np.array(true_scores) - np.array(data_matrix[np.where(labels==label_id)[0]].mean(axis=0)))**2)
