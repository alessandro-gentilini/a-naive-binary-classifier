import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

mu_1 = 1
sigma_1 = 1
mu_2 = 7*mu_1
sigma_2 = sigma_1
N_1 = 500
N_2 = 110
c1 = np.random.normal(mu_1,sigma_1,N_1)
c2 = np.random.normal(mu_2,sigma_2,N_2)
c3 = np.concatenate((c1,c2))
c3.sort()
d=np.diff(c3,n=1)
arg_max=np.argmax(d)
if arg_max+1>=c3.size:
	raise Exception('unexpected')

threshold=(c3[arg_max]+c3[arg_max+1])/2
print(arg_max)
print(threshold)

c1_error = 0
for x in c1:
	if x>=threshold:
		c1_error=c1_error+1

if c1_error>0:
	print('nbc misclassified c1')		
	print(100*c1_error/(1.0*c1.size))

c2_error = 0
for x in c2:
	if x<threshold:
		c2_error=c2_error+1

if c2_error>0:
	print('nbc misclassified c2')		
	print(100*c2_error/(1.0*c2.size))

c4 = np.concatenate((c1,c2))
kmeans = KMeans(n_clusters=2, random_state=0).fit(c4.reshape(-1,1))	

c1_estimated = kmeans.labels_[0:c1.size-1]
c2_estimated = kmeans.labels_[c1.size:c1.size+c2.size-1]

c1_0 = (c1_estimated==0).sum()
c1_1 = (c1_estimated==1).sum()
if min(c1_0,c1_1)!=0:
	print('KMeans misclassified c1')
	print(100*min(c1_0,c1_1)/(1.0*(c1_0+c1_1)))

c2_0 = (c2_estimated==0).sum()
c2_1 = (c2_estimated==1).sum()
if min(c2_0,c2_1)!=0:
	print('KMeans misclassified c2')
	print(100*min(c2_0,c2_1)/(1.0*(c2_0+c2_1)))



plt.plot(d)
plt.plot(c3)
plt.plot(c1)
plt.plot(c2)
plt.plot(np.repeat(threshold,c3.size))
plt.show()


