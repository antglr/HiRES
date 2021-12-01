import sys
import numpy as np

BUF_SZ = 16384

def postp(fin, nch=1):
    pts_per_ch = BUF_SZ // nch

    pvs = {}
    for ll in fin.readlines():
        a = ll.split()
        ta = [float(x) for x in a[2].split(":")]
        #t = (ta[0]*60 + ta[1])*60 + ta[2]
        # print(a[0], a[1], a[2], a[3], t)
        pv = a[0]
        npt = int(a[3])
        v = [float(x) for x in a[3:]][0:pts_per_ch]
        if abs(npt-pts_per_ch) > 1:
            print("%d != %d" % (npt, pts_per_ch))
            exit(-1)
        if pv not in pvs:
            print("New PV: %s" % pv)
            pvs[pv] = []
        pvs[pv].extend(v)
    sk = sorted(pvs.keys())
    hdr = " ".join([pv for pv in sk])
    print(hdr)
    pvl = [pvs[pv] for pv in sk]
    d = np.vstack(pvl)
    print(d.shape)
    return d.transpose(), hdr

if __name__ == "__main__":
    fname = sys.argv[1]
    nch = int(sys.argv[2])*2  # I+Q
    with open(fname, 'r') as FH:
        d, hdr = postp(FH, nch)
    np.savetxt(fname + ".dat", d, fmt="%9.5f", header=hdr)
