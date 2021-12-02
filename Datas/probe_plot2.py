import sys
import numpy as np
from matplotlib import pyplot as plt

wsp = 4
dt = wsp*22/(102.14e6)
NCH = 5*2  # 5*2 channels enabled

fname = sys.argv[1]
x = np.loadtxt(fname)

x_cav_main = x[:,0]  # 0 - CAV_MAIN AMP
x_cavp_main = x[:,1]
x_cav_aux  = x[:,2]  # 2 - CAV_AUX AMP
x_cavp_aux = x[:,3]
x_laser_amp = x[:,4] # 4 - Laser AMP
x_laser_phs = x[:,5] # 5 - Laser PHS
row = x_cav_main.shape[0]

# Compute energy gain according to E = A*cos(phi)
x_egain_main = x_cav_main*np.cos(x_cavp_main*np.pi/180.0)
x_egain_aux = x_cav_aux*np.cos((x_cavp_aux+79.0)*np.pi/180.0)

# plot_train = True
# if plot_train:
#     t = np.arange(0, row)*dt
#     plt.subplot("311")
#     plt.plot(t, x_cav_main/max(x_cav_main), label="In-loop Magnitude")
#     plt.plot(t, x_cav_aux/max(x_cav_aux), label="Out-of-loop Magnitude")
#     plt.legend()
#     plt.subplot("312")
#     plt.plot(t, x_egain_main/max(x_egain_main), label="In-loop Energy gain")
#     plt.plot(t, x_egain_aux/max(x_egain_aux), label="Out-of-loop Energy gain")
#     plt.legend()
#     plt.subplot("313")
#     plt.plot(t, x_cavp_main/max(x_cavp_main), label="In-loop Phase")
#     plt.plot(t, x_cavp_aux/max(x_cavp_aux), label="Out-of-loop Phase")
#     plt.legend()
#     plt.suptitle(fname + "\nCavity probes normalized to max")
#     plt.show()

buf_pts = int(16384/NCH)
n_bufs  = int(row/buf_pts)
print(buf_pts, n_bufs, row)

# slice into buffers
x_cav_main = np.reshape(x_cav_main, (n_bufs, buf_pts))
x_cavp_main = np.reshape(x_cavp_main, (n_bufs, buf_pts))
x_egain_main = np.reshape(x_egain_main, (n_bufs, buf_pts))

x_cav_aux = np.reshape(x_cav_aux, (n_bufs, buf_pts))
x_cavp_aux = np.reshape(x_cavp_aux, (n_bufs, buf_pts))
x_egain_aux = np.reshape(x_egain_aux, (n_bufs, buf_pts))

x_laser_amp = np.reshape(x_laser_amp, (n_bufs, buf_pts))
x_laser_phs = np.reshape(x_laser_phs, (n_bufs, buf_pts))

# do_histogram = True
# if do_histogram:
#     sample_t = 580e-6
#     sample_pt = int(sample_t / dt)
#     bins = 100

#     cava_main_pt = x_cav_main[:, sample_pt]
#     cava_aux_pt = x_cav_aux[:, sample_pt]
#     cavp_main_pt = x_cavp_main[:, sample_pt]
#     cavp_aux_pt = x_cavp_aux[:, sample_pt]
#     plt.subplot("221")
#     y = cava_main_pt
#     plt.hist(y, bins=bins, label="Mean: %5.3f, Std: %5.3e" % (np.mean(y), np.std(y)))
#     plt.legend(prop={"size":7})
#     plt.title("CAV Main MAG")
#     plt.subplot("222")
#     y = cavp_main_pt
#     plt.hist(y, bins=bins, label="Mean: %5.3f, Std: %5.3e" % (np.mean(y), np.std(y)))
#     plt.legend(prop={"size":7})
#     plt.title("CAV Main PH")
#     plt.subplot("223")
#     y = cava_aux_pt
#     plt.hist(y, bins=bins, label="Mean: %5.3f, Std: %5.3e" % (np.mean(y), np.std(y)))
#     plt.legend(prop={"size":7})
#     plt.title("CAV Aux MAG")
#     plt.subplot("224")
#     y = cavp_aux_pt
#     plt.hist(y, bins=bins, label="Mean: %5.3f, Std: %5.3e" % (np.mean(y), np.std(y)))
#     plt.legend(prop={"size":7})
#     plt.title("CAV Aux PH")
#     plt.suptitle("MAG/PH histogram for time %5.3f us over %d pulses [%s]" % (sample_t*1e6, n_bufs, fname))
#     plt.show()


# Average over X point window during flat-top
win = int(100e-6/ dt)
win_start = int(580e-6 / dt)
xcavm_avg = np.mean(x_cav_main[:,win_start:win_start+win], axis=1)
xcavmp_avg = np.mean(x_cavp_main[:,win_start:win_start+win], axis=1)
xegainm_avg = np.mean(x_egain_main[:,win_start:win_start+win], axis=1)

xcava_avg = np.mean(x_cav_aux[:,win_start:win_start+win], axis=1)
xcavap_avg = np.mean(x_cavp_aux[:,win_start:win_start+win], axis=1)
xegaina_avg = np.mean(x_egain_aux[:,win_start:win_start+win], axis=1)

x_laser_amp_avg = np.mean(x_laser_amp[:,win_start:win_start+win], axis=1)
x_laser_phs_avg = np.mean(x_laser_phs[:,win_start:win_start+win], axis=1)

# Normalize only amplitude and energy gain
xcavm_ac = (xcavm_avg-np.mean(xcavm_avg))/np.mean(xcavm_avg)
xcavmp_ac = (xcavmp_avg-np.mean(xcavmp_avg))
xegainm_ac = (xegainm_avg-np.mean(xegainm_avg))/np.mean(xegainm_avg)

xcava_ac = (xcava_avg-np.mean(xcava_avg))/np.mean(xcava_avg)
xcavap_ac = (xcavap_avg-np.mean(xcavap_avg))
xegaina_ac = (xegaina_avg-np.mean(xegaina_avg))/np.mean(xegaina_avg)

x_laser_amp_ac = (x_laser_amp_avg-np.mean(x_laser_amp_avg))/np.mean(x_laser_amp_avg)
x_laser_phs_ac = (x_laser_phs_avg-np.mean(x_laser_phs_avg))/np.mean(x_laser_phs_avg)

# plt.subplot("311")
# plt.plot(xcavm_ac, label='In-loop Magnitude normalized error')
# plt.plot(xcava_ac, label='Out-of-loop Magnitude normalized error')
# plt.legend(loc="upper right")
# plt.subplot("312")
# plt.plot(xegainm_ac, label='In-loop Energy normalized error')
# plt.plot(xegaina_ac, label='Out-of-loop Energy normalized error')
# plt.legend(loc="upper right")
# plt.subplot("313")
# plt.plot(xcavmp_ac, label='In-loop Phase error')
# plt.plot(xcavap_ac, label='Out-of-loop Phase error')
# plt.legend(loc="upper right")
# plt.suptitle(fname + " - " + ("%5.3f" % (dt*win*1e6)) + " us flat-top average over time")
# plt.show()

# Write out data
if True:
    save_d = [("In-loop Magnitude", xcavm_ac), ("In-loop Phase", xcavmp_ac),
              ("OOL Magnitude", xcava_ac), ("OOL Phase", xcavap_ac),
              ("In-loop Energy", xegainm_ac), ("OOL Energy", xegaina_ac),
            ("Laser Amplitude", x_laser_amp_ac), ("Laser Phase", x_laser_phs_ac)]

    header = ",".join([x[0] for x in save_d])
    data = tuple([x[1] for x in save_d])
    np.savetxt(sys.argv[1]+".post", np.vstack(data).transpose(), header=header)
    np.savetxt("RF.post", np.vstack(data).transpose(), header=header)

print("Hi")