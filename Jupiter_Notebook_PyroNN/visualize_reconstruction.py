import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import numpy as np

def show_reco_views(reco,geometry):
    plt.figure(figsize=(12, 4))
    
    # Y view
    plt.subplot(1, 3, 1)
    plt.imshow(np.squeeze(torch.squeeze(reco, dim=0).cpu().detach().numpy())[:, geometry.volume_shape[0] // 2, :], cmap='gray')
    plt.axis('on')
    plt.title('YZ View')
    
    # X view
    plt.subplot(1, 3, 2)
    plt.imshow(np.squeeze(torch.squeeze(reco, dim=0).cpu().detach().numpy())[geometry.volume_shape[0] // 2, :, :], cmap='gray')
    plt.axis('on')
    plt.title('XY View')
    
    # Z view
    plt.subplot(1, 3, 3)
    plt.imshow(np.squeeze(torch.squeeze(reco, dim=0).cpu().detach().numpy())[:, :, geometry.volume_shape[0] // 2], cmap='gray')
    plt.axis('on')
    plt.title('XZ View')
    
    plt.tight_layout()
    plt.show()