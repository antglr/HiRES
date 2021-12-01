import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
#Antonio 
#from jitter_rms import jitter_rms
import numpy as np
from matplotlib import pyplot as plt

def plot_stack(x, yy, plot_max, xlabel, title='', line='-'):
    for i in range(0, min(np.shape(yy)[0], plot_max)):
        plt.plot(x, yy[i], line)
    plt.xlabel(xlabel)
    plt.title(title)

wsp = 4
dt = wsp*22/(102.14e6)

# x = np.loadtxt("sync_acq_wsp4_5ch_1hz_ucam1.rf.dat")
x = np.loadtxt(sys.argv[1])
NROW = np.shape(x)[0]
NCH = 5*2  # 5*2 channels enabled

# 0 = Main cavity probe
# 2 = Aux cavity probe
CAV = 0
x_amp = x[:,CAV]
x_pha = x[:,CAV+1]

buf_pts = int(16384/NCH)
n_bufs  = int(NROW/buf_pts)
print(NROW, buf_pts, n_bufs)

# slice into buffers
amp = np.reshape(x_amp, (n_bufs, buf_pts))
pha = np.reshape(x_pha, (n_bufs, buf_pts))

# cut to integer number of pulse periods (1.5 ms); assume that there's at least
# one full pulse period in each buffer
nsamp = int(1.4e-3/dt)
npulse = buf_pts//nsamp
amp = amp[:,0:nsamp*npulse]
pha = pha[:,0:nsamp*npulse]
if True:
    t = np.arange(0, nsamp*dt, dt)*1e3
    plot_stack(t, amp, 20, 'time [ms]', "Stacked pulse amplitude")
    plt.show()
    plot_stack(t, pha, 20, 'time [ms]', "Stacked pulse phase")
    plt.show()

# slice all flat-tops into buffers
START_P = 0.580e-3
STOP_P = START_P + 0.100e-3
start = int(START_P/dt)
stop = int(STOP_P/dt)
period = int(1.4e-3/dt)
nwave = npulse*n_bufs
nsamp = stop-start
slice_amp = np.zeros((nwave, nsamp))
slice_pha = np.zeros((nwave, nsamp))
nn = 0
for i in range(n_bufs):
    for j in range(npulse):
        slice_amp[nn] = amp[i][start + j*period: stop + j*period]
        slice_pha[nn] = pha[i][start + j*period: stop + j*period]
        nn += 1
#Antonio
#amp_jitter, aj = jitter_rms(slice_amp, norm=True, verbose=True)
#pha_jitter, pj = jitter_rms(slice_pha, norm=False, verbose=True)

if True:
    t = np.arange(0, nsamp*dt, dt)*1e3
    title = "Stacked closed-loop flat-top %s " + "(%4.2f to %4.2f ms)" % (START_P*1e3, STOP_P*1e3)
    plot_stack(t, slice_amp, 10, 'time [ms]', title % "amplitude", "-.")
    plt.plot(t, aj, 'b-', label="RMS error (norm. jitter = %4.2e" % amp_jitter)
    plt.legend(loc = "upper right")
    plt.show()
    plot_stack(t, slice_pha, 10, 'time [ms]', title % "phase", "-.")
    plt.plot(t, pj, 'b-', label="RMS error (jitter = %4.2e" % pha_jitter)
    plt.legend(loc = "upper right")
    plt.show()
