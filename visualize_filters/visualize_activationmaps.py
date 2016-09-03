import pdb
import numpy as np

import matplotlib.pyplot as plt


for i in range(1,11,1):
    print 'loc{} | visualizing activation maps..'.format(i)
    
    fname = "activationmaps_npy/loc{0:d}.npy".format(i)
    activationmaps = np.load(fname)

    fig, axs = plt.subplots(1,5,figsize=(10,4),facecolor='w',edgecolor='k')
    axs = axs.ravel()
    for l in range(5):
        axs[l].imshow(activationmaps[l])
        axs[l].get_xaxis().set_ticks([])
        axs[l].get_yaxis().set_ticks([])
        if l == 0:
            axs[l].set_title('CF1')
        elif l == 1:
            axs[l].set_title('CF2')
        elif l == 2:
            axs[l].set_title('CF3')
        elif l == 3:
            axs[l].set_title('Cat')
        elif l == 4:
            axs[l].set_title('Output')

    plt.tight_layout()
    # save
    fname = "activationmaps_results/loc{}.png".format(i)
    plt.savefig(fname)
    #plt.show()

