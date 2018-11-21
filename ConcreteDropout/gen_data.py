import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

N = 100000 # the number of sample will be generated
noise_list = [0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0]
for noise in noise_list:
    X = np.random.randn(N,1).flatten()
    e = np.random.normal(0.0, noise, N)
    Y = 2.*X + 8. + e
    Y1 = 2.*X + 8. 

    data_ = [X,Y]
    data_ = np.asarray(data_)
    np.save('./data/synthetic_'+str(noise)+'.npy', data_)
