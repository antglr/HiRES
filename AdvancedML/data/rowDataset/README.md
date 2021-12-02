# 2021-06-29 Test Session Data

Obtained with the following camonitor command:
camonitor TST:LLRF:ODATA:{CAV,FWD2,LASER1,SPARE_1300}:{A,P}WF UCam1:image2:ArrayData LCam1:image2:ArrayData > [filename]

- Open loop capture:
   - 1m30s test capture: OK (sync_acq_wsp4_5ch_1hz_ucam1_lcam1)
   - 20m0s capture: OK (sync_acq_wsp4_5ch_1hz_ucam1_lcam1_long)
   - Buncher off
   - 7818 / 3x2 = 1303 RF buffers
   - 1303 ucam buffers

- Close loop capture:
   - 20m0s capture: OK (sync_acq_wsp4_5ch_1hz_ucam1_lcam1_cloop_long)
   - Buncher off
   - 7818 / 6 = 1303 RF buffers
   - 1303 ucam buffers

- Close loop + buncher OL capture:
   - 20m0s capture: OK (sync_acq_wsp4_5ch_1hz_ucam1_lcam1_cloop_buncher_ol_long)
   - Added SPARE_1300 signal; buncher probe
   - Buncher on but in open-loop
   - LCam1 didn't update; only captured one shot
   - 11368 / 8 = 1421 RF buffers
   - 1421 ucam buffers

- Close loop + buncher CL capture:
   - Boosted buncher probe by 10x
   - 9m10s capture failed due to dropped frame (sync_acq_wsp4_5ch_1hz_ucam1_lcam1_cloop_buncher_cl_long)
   - 4720 / 8 = 590 RF buffers
   - 592 ucam buffers
   - RF is okay up until buffer 580; truncate both cam and rf to 580 entries

   - Reduced trigger rate to ~0.8 Hz (sync_acq_wsp4_5ch_08hz_ucam1_lcam1_cloop_buncher_cl_long)
   - RF tripped after 20 min, might have to discard last buffers but still
     check that we have matching numbers
   - LCam1 didn't update; only captured one shot
   - 8560 / 8 = 1070 RF buffers
   - 1071 ucam buffers; delete the last
   - Cam acquisition period = 1.16 seconds
   - RF acquisition period = 1.16 seconds but very jittery

Pre-processing to convert raw data to columnar format:
grep UCam1 fname > fname.ucam1
grep LCam1 fname > fname.lcam1
grep LLRF fname > fname.lcam1

synchronous acquisition at 1-1.16 Hz with 5 RF channels enabled on the LLRF system and a decimation
rate (wsp) of 4 (1 is the minimum, 255 is the maximum). Each line in the file corresponds to an
individual PV:

TST:LLRF:ODATA:CAV:AWF -> Main Cavity Amplitude
TST:LLRF:ODATA:CAV:PWF -> Main Cavity Phase
TST:LLRF:ODATA:FWD2:AWF -> Auxiliary Cavity Amplitude
TST:LLRF:ODATA:FWD2:PWF -> Auxiliary Cavity Phase
TST:LLRF:ODATA:LASER1:AWF -> Laser Amplitude
TST:LLRF:ODATA:LASER1:PWF -> Laser Phase
TST:LLRF:ODATA:SPARE_1300:AWF -> Buncher cavity amplitude
TST:LLRF:ODATA:SPARE_1300:PWF -> Buncher cavity phase
UCam1:image2:ArrayData -> Camera image from viewscreen before second dipole
LCam1:image2:ArrayData -> Not triggered; free-running

Files with a .dat extension have gone through a basic pre-processing stage that converts the raw data
into a columnar format that can be loaded directly into Python/Matlab/Octave:
python3 postp.py fname.rf 5
python3 postp_cam.py fname.ucam1

## Camera acquisition settings

Camera acquisition settings were as follows:
- UCam1: 200x125
- LCam1: 60x80 (free-running, not triggered)

## RF acquisition settings

The LLRF system uses a 16384-long memory to record waveforms. This space is divided by the number
of acquisition channels that are enabled at a time. With 5 channels enabled, each gets 16384 // (5x2)
points per buffer, where the factor of 2 accounts for IQ storage.

The acquisition timestep can be as fast as 22/102.14e6 = 0.215 us if the decimation rate (wsp) is set
to 1. For a decimation rate of 5, the timestep is 22x5/102.14e6 = 1.077 us

## Example scripts

The enclosed Python scripts can be used to perform basic analysis on the data. Example command-lines
follow:

python3 beamshow.py sync_acq_wsp4_5ch_1hz_ucam1_lcam1_long.ucam1.dat 200 125

python3 jitter_ap.py sync_acq_wsp4_5ch_1hz_ucam1_lcam1_long.rf.dat

python3 probe_plot.py sync_acq_wsp4_5ch_1hz_ucam1_lcam1_long.rf.dat

python3 rf_cam_corr.py sync_acq_wsp4_5ch_1hz_ucam1_lcam1_long.rf.dat.post sync_acq_wsp4_5ch_1hz_ucam1_lcam1_long.ucam1.dat.post

For maximum correlation, it might be necessary to slide RF and beam data by up to 3 buffers; see
rf_cam_corr.py for how to do this
