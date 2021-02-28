import matplotlib.pyplot as plt
import numpy as np

x=['0','0.25','0.5','0.75','1']
cora = [5.72, 4.72, 3.37, 2.62, 2.31]
citeseer = [5.09, 4.09, 3.19, 2.12, 1.59]
pubmed = [2.51, 1.81, 1.24, 1.07, 1]
fig=plt.figure()
fig.show()
ax=fig.add_subplot(111)

ax.plot(x,cora,c='r',marker="o",markersize=5, ls='-',label='Cora')
ax.plot(x,citeseer,c='g',marker="s",ls='-',label='Citeseer')
ax.plot(x,pubmed,c='b',marker="^",ls='-',label='Pubmed')
ax.set_xlabel("Ratio of Cold Start")
ax.set_ylabel("Accuracy")
plt.legend(loc=1)
plt.draw()
