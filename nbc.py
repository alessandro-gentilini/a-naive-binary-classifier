import numpy as np
import matplotlib.pyplot as plt

mu_1 = 1
sigma_1 = .5
mu_2 = 5*mu_1
sigma_2 = sigma_1
N_1 = 10
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
print(threshold)


plt.plot(d)
plt.plot(c3)
plt.show()


