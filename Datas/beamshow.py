from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import norm
import sys

img = sys.argv[1]  # "ucam1_beam_vfast_200_400.dat"
[x,y] = sys.argv[2], sys.argv[3] # 400, 200
max_img = -1
if len(sys.argv) > 4:
    max_img = int(sys.argv[4])

x, y = int(x), int(y)

if max_img > 0:
    im_a = np.loadtxt(img, max_rows=(x*y*max_img))
else:
    im_a = np.loadtxt(img)

num_img = len(im_a)//(x*y)

im_xy = im_a.reshape((num_img, y,x))

im_x_mean = np.zeros((num_img, x))
WIN = 2
im_x_smooth = np.zeros((num_img, x-WIN+1))
im_x_center = np.zeros((num_img, 1))
xP = np.zeros((num_img, 1))
im_x_std = np.zeros((num_img, 1))
for i in range(num_img):
    im_x_mean[i] = np.sum(im_xy[i], 0)  # Project onto X
    # Moving average filter
    im_x_smooth[i] = np.convolve(im_x_mean[i], np.ones(WIN), 'valid') / WIN

    # Gaussian fit
    # Convert from histogram-like to raw data:
    edges = np.arange(im_x_mean[i].shape[0]+1)*1.0
    x = np.repeat(edges[:-1], im_x_mean[i].astype("int64"))
    mean, std =norm.fit(x)
    im_x_center[i] = mean
    im_x_std[i] = std
    pX = x / np.sum(x) 
    xP[i] = np.sum( x * pX)
    # Max fit
    #im_x_center[i] = np.argmax(im_x_smooth[i])

# for i in range(10):
#     ax1 = plt.subplot(2, 1, 1)
#     ax2 = plt.subplot(2, 1, 2)
#     ax1.imshow(im_xy[i,:,:])
#     ax2.plot(im_x_mean[i])
#     ax2.vlines(x = im_x_center[i], ymin = 0, ymax = np.max(im_x_mean),color='b',label="Fit")
#     ax2.vlines(x = xP[i], ymin = 0, ymax = np.max(im_x_mean),color='r',label="Proj")
#     plt.legend()
#     plt.title(img + " Image No: " + str(i))
#     plt.show()

#Fit
x_scaled = 718 - im_x_center * 0.027  # KeV/pixel
mean_val = np.mean(x_scaled)
err = x_scaled - mean_val
rms = np.sqrt(np.mean((err)**2))
jitter_norm = rms/mean_val
#Proj
xP_scaled = 718 - xP * 0.027  # KeV/pixel
mean_valP = np.mean(xP_scaled)
errP = xP_scaled - mean_valP
rmsP = np.sqrt(np.mean((errP)**2))
jitter_normP = rmsP/mean_valP


#plt.plot(err/mean_val, label="RMS jitter norm=%4.2e" % jitter_norm)
# plt.title("Centroid error")
# plt.legend(loc="upper right")
# plt.xlabel("Acquisition #")
# plt.ylabel("keV")
# plt.show()


if True:
    save_d = [("Centroid error", err/mean_val)]
    save_dP = [("Centroid error", errP/mean_valP)]

    header = ",".join([x[0] for x in save_d])
    data = tuple([x[1] for x in save_d])
    np.savetxt(sys.argv[1]+".post", np.vstack(data).transpose(), header=header)
    np.savetxt("CenterError_Fit.post", np.vstack(data).transpose(), header=header)
    headerP = ",".join([x[0] for x in save_dP])
    dataP = tuple([x[1] for x in save_dP])
    np.savetxt("CenterError_Proj.post", np.vstack(dataP).transpose(), header=headerP)
