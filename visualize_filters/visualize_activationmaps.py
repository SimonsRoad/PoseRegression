import pdb
import numpy as np

import matplotlib.pyplot as plt


# for 10 locations
for i in range(1,11,1):
    print 'loc{} | visualizing activation maps..'.format(i)
    
    fname = "activationmaps_npy/loc{0:d}.npy".format(i)
    activationmaps = np.load(fname)
    numlayers = activationmaps.shape[0]
    nameoflayers = ['Res1', 'Res2', 'Res3', 'Res4', 'CF1', 'CF2', 'CF3', 'Cat', 'Output']

    fig, axs = plt.subplots(1,numlayers,figsize=(12,3),facecolor='w',edgecolor='k')
    axs = axs.ravel()
    for l in range(numlayers):
        axs[l].imshow(activationmaps[l])
        axs[l].get_xaxis().set_ticks([])
        axs[l].get_yaxis().set_ticks([])
        axs[l].set_title(nameoflayers[l])
    plt.tight_layout()
    # save
    fname = "activationmaps_results/loc{}.png".format(i)
    plt.savefig(fname)
    #plt.show()

