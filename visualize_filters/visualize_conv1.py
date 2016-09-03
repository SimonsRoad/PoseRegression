import pdb
import numpy as np

import matplotlib.pyplot as plt


for i in range(1,11,1):
    print 'loc{} | visualizing conv1 filters..'.format(i)
    
    fname = "conv1_npy/conv1_weights_loc{}.npy".format(i)
    conv1 = np.load(fname)
    conv1 = np.transpose(conv1, (0,2,3,1))

    fig, axs = plt.subplots(8,8,figsize=(8,8),facecolor='w',edgecolor='k')
    axs = axs.ravel()
    for j in range(64):
        axs[j].imshow(conv1[j])
        axs[j].get_xaxis().set_ticks([])
        axs[j].get_yaxis().set_ticks([])
    plt.show()
    pdb.set_trace()

    # save
    fname = "conv1_results/conv1_weights_loc{}.png".format(i)
    #plt.savefig(fname)
    

