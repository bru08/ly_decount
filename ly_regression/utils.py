import numpy as np
import matplotlib.pyplot as plt
# %%
def mat_gauss(ker=5, var=1):
    mat = np.ones((ker,ker))
    mi = ker // 2

    for i in range(ker):
        for j in range(ker):
            mean_dist =  np.sqrt( (j - mi) ** 2 + (i - mi) ** 2 )
            coef = 2 * np.pi * var
            exp_coef = - 1 / (2 * var) 
            p = coef * np.exp(exp_coef * (mean_dist ** 1.8))
            mat[i,j] *= p
    return mat
# %%
def count_dens_mask(coords, img_size=(199,199), var=12, spot_size=37):
    img = np.zeros(img_size)
    spot = mat_gauss(spot_size, var)
    sshape = spot.shape
    for pt in coords:
        x_slice = slice(pt[0]-sshape[0]//2, pt[0]+sshape[0]//2 + 1)
        y_slice = slice(pt[1]-sshape[0]//2, pt[1]+sshape[1]//2 + 1)
        img[x_slice, y_slice] += spot
    img /= np.max(img)
    return img
a = count_dens_mask([(50,100), (50,120)], var=30, spot_size=47)
plt.imshow(a)
# %%
