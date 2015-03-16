import pylab
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np

pylab.clf()
plt.figure(figsize=(10, 5), dpi=80)

x=np.linspace(-1,1,100)
y= 1.0 - stats.logistic.cdf(x, 0, .2)
plt.subplot(1,2,2)
plt.axvline(-.6,0,1,color='.3',linestyle='--', linewidth=2)
plt.axvline(.6,0,1,color='.3',linestyle=':', linewidth=2)
plt.plot(x,y,color='.2', linewidth=3)
plt.hold
plt.xlabel('Evaluator Expertise $\mathbf{a_{p}}$ - Design Difficulty $\mathbf{d_{d}}$', fontsize=16)
plt.ylabel(r'Evaluation Error  $\sigma_{pd}^2$', fontsize=16)

plt.legend(('Low Expertise', 'High Expertise'),
          'upper right', shadow=True, fancybox=True)
leg = plt.gca().get_legend()
frame  = leg.get_frame()  # the patch.Rectangle instance surrounding the legend
frame.set_facecolor('0.93')      # set the frame face color to light gray

plt.subplot(1,2,1)
scale = .8
scale2 = .05
x=np.linspace(0,1,1000)
y1=np.array(stats.truncnorm.pdf(x,-.7/scale,.3/scale,.7,scale))
y2=np.array(stats.truncnorm.pdf(x,-.7/scale2,.3/scale2,.7,scale2))

#------- PLOT Error Variance with Truncated Normal ---------
plt.plot(x,y1,color='.3',linestyle='--', linewidth=2)
plt.plot(x,y2,color='.3',linestyle=':', linewidth=2)
plt.axvline(.7,0,1,color='.2',linestyle='-', linewidth=1)

plt.xlabel(r'Evaluation $\mathbf{r_{pd}}$', fontsize=16)
plt.ylabel(r'Probabilty of Evaluation', fontsize=16)
plt.xlim([0,1])

plt.legend(('Low Expertise', 'High Expertise','True Score'),
          'upper left', shadow=True, fancybox=True)
leg = plt.gca().get_legend()
frame  = leg.get_frame()  # the patch.Rectangle instance surrounding the legend
frame.set_facecolor('0.93')      # set the frame face color to light gray


pylab.setp(plt.subplot(1,2,1).get_yticklabels(), visible=False)
pylab.setp(plt.subplot(1,2,2).get_yticklabels(), visible=False)
plt.tight_layout()

pylab.savefig('logistic_relation.png')

