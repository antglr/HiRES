from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import norm
import sys

img = "C:/Users/ecropp/Desktop/Berkeley/2021_06_29_RF_Data/20210629_data_full/sync_acq_wsp4_5ch_1hz_ucam1_lcam1_long.ucam1.dat"  # "ucam1_beam_vfast_200_400.dat"
[x,y] =200,125 # 400, 200
max_img = -1
# if len(sys.argv) > 4:
#     max_img = int(sys.argv[4])

x, y = int(x), int(y)

if max_img > 0:
    im_a = np.loadtxt(img, max_rows=x*y*max_img)
else:
    im_a = np.loadtxt(img)

num_img = len(im_a)//(x*y)

im_xy = im_a.reshape((num_img, y,x))

im_x_mean = np.zeros((num_img, x))
WIN = 2
im_x_smooth = np.zeros((num_img, x-WIN+1))
im_x_center = np.zeros((num_img, 1))
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

    # Max fit
    #im_x_center[i] = np.argmax(im_x_smooth[i])

for i in range(3):
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)
    ax1.imshow(im_xy[i,:,:])
    ax2.plot(im_x_mean[i], label="Center pix = %f, Std = %f" % (im_x_center[i], im_x_std[i]))
    plt.legend()
    plt.title(img + " Image No: " + str(i))
    plt.show()

x_scaled = 718 - im_x_center * 0.027  # KeV/pixel
mean_val = np.mean(x_scaled)
err = x_scaled - mean_val
rms = np.sqrt(np.mean((err)**2))
jitter_norm = rms/mean_val
plt.plot(err/mean_val, label="RMS jitter norm=%4.2e" % jitter_norm)
plt.title("Centroid error")
plt.legend(loc="upper right")
plt.xlabel("Acquisition #")
plt.ylabel("keV")
plt.show()

if True:
    save_d = [("Centroid error", err/mean_val)]

    header = ",".join([x[0] for x in save_d])
    data = tuple([x[1] for x in save_d])
    np.savetxt(sys.argv[1]+".post", np.vstack(data).transpose(), header=header)
