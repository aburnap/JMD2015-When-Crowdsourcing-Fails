#-----------------------------------------------------------------------------
#      
#       Paper: When Crowdsourcing Fails: A Study of Expertise on Crowdsourced 
#              Design Evaluation
#       Author: Alex Burnap - aburnap@umich.edu
#       Date: October 10, 2014
#       License: Apache v2
#       Description: Plotting code for simulated hetereogenous crowd
#
#-----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import pylab
import csv
pylab.clf()

mean_vector = []
majority_vote_MSqE = []
estimated_MSqE = []

with open("results_homo_mean.csv","rb") as csvfile:
        results=csv.reader(csvfile, dialect='excel')
        for row in results:
		try:
			mean_vector.append(row[0])
			majority_vote_MSqE.append(row[1])
			estimated_MSqE.append(row[2])
		except IndexError:
			pass

plt.scatter(mean_vector, majority_vote_MSqE, color='gray',marker='o')
plt.scatter(mean_vector, estimated_MSqE, color='black', marker = '*', s=80)

plt.grid(True)
plt.xlabel('Average Evaluator Expertise', fontsize=15)
plt.ylabel('Design Evaluation Error', fontsize=15)
pylab.xlim([0,1])
pylab.ylim([-.01,1.7])
fig=pylab.gca()
pylab.setp(fig.get_yticklabels(), visible=False)
plt.legend(('Averaging', 'Bayesian Network'),
           'upper right', shadow=True, fancybox=True)
leg = plt.gca().get_legend()
frame  = leg.get_frame()  # the patch.Rectangle instance surrounding the legend
frame.set_facecolor('0.93')      # set the frame face color to light gray
plt.title('Homogeneous Crowds with Different Average Expertises', fontsize=18)
pylab.savefig('error_homo_mean.png')
