import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pylab
import scipy.stats as stats

pylab.clf()
# Plot between -10 and 10 with .001 steps.
x = np.arange(0, 1, 0.001)
# Mean = 0, SD = 2.

plt.subplot(2,2,1)
plt.xlabel('Evaluator Expertise $\mathbf{a_{p}}$', fontsize=14)
plt.ylabel('Proportion of Crowd', fontsize=14)
plt.title('Case I: Homogenous Crowds')
plt.plot(x, norm.pdf(x,.8,.02), linewidth=2, color='gray',linestyle='--')
plt.plot(x, norm.pdf(x,.3,.02), linewidth=2, color='gray',linestyle=':')
plt.plot(x, norm.pdf(x,.5,.02), linewidth=2, color='gray')

plt.subplot(2,2,2)
plt.xlabel('Evaluator Expertise $\mathbf{a_{p}}$', fontsize=14)
plt.ylabel('Proportion of Crowd', fontsize=14)
plt.title('Case II: Heterogeneous Crowds')
scale=.1
y1=np.array(stats.truncnorm.pdf(x,-.2/scale,.8/scale,.2,scale))
scale=.4
y2=np.array(stats.truncnorm.pdf(x,-.2/scale,.8/scale,.2,scale))
plt.plot(x, norm.pdf(x,.2,.2), linewidth=2, color='gray',linestyle=':')
plt.plot(x, y1, linewidth=2, color='gray',linestyle='--')
plt.plot(x, y2, linewidth=2, color='gray')

pylab.setp(plt.subplot(2,2,1).get_yticklabels(), visible=False)
pylab.setp(plt.subplot(2,2,2).get_yticklabels(), visible=False)
plt.tight_layout()
pylab.savefig('crowd_expertises.png')
