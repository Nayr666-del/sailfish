import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram
import argparse
import pickle as pk
import sys

class FixNumpyCoreUnpickler(pk.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)
    

def load_checkpoint(filename, require_solver=None):
    with open(filename, "rb") as f:
        chkpt = FixNumpyCoreUnpickler(f).load()
    return chkpt

class DavidTimeseries:
    def __init__(self, Checkpoint):
        #Checkpoint = load_checkpoint(chkpt)
        timeseries_data = Checkpoint['timeseries']
        max_len         = max(len(arr) for arr in timeseries_data)
        ts              = np.array([np.pad(arr, (0, max_len - len(arr)), 'constant') for arr in timeseries_data])

        self.pointmasses     = Checkpoint["point_masses"]
        self.currenttime     = Checkpoint["time"] / 2 / np.pi 
        self.modelparams     = Checkpoint['model_parameters'] 
        self.time            = np.array([s[ 0] for s in ts])
        self.semimajor_axis  = np.array([s[ 1] for s in ts])
        self.eccentricity    = np.array([s[ 2] for s in ts])
        self.density_floor   = np.array([s[ 3] for s in ts])
        self.pressure_floor  = np.array([s[ 4] for s in ts])
        self.Accreted_energy = np.array([s[ 5] for s in ts])

        self.optical         = np.array([s[ 6] for s in ts])
        self.infared         = np.array([s[ 7] for s in ts])
        self.bolometric      = np.array([s[ 8] for s in ts])
        self.uncounted_cells = np.array([s[ 9] for s in ts])

        self.mdot1           = np.array([s[10] for s in ts])
        self.mdot2           = np.array([s[11] for s in ts])
        self.torque_g        = np.array([s[12] for s in ts])
        self.torque_a        = np.array([s[13] for s in ts])
        self.power_g1        = np.array([s[14] for s in ts])
        self.power_g2        = np.array([s[15] for s in ts])
        self.power_a1        = np.array([s[16] for s in ts])
        self.power_a2        = np.array([s[17] for s in ts])
        self.jdisk           = np.array([s[18] for s in ts])
        self.uv              = np.array([s[19] for s in ts])
        self.xray            = np.array([s[20] for s in ts])
        self.Max_temp        = np.array([s[21] for s in ts])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", type=str, nargs="+")
    parser.add_argument(
        "--Accretion",
        "-a",
        action='store_true',
        help="whether to plot the binary's accretion timeseries",
    )
    parser.add_argument(
        "--Lightcurves",
        "-lc",
        action='store_true',
        help="whether to plot the optical and infared luminosities of the disk",
    )
    args = parser.parse_args()
    filename1 = args.checkpoints[0]
    filename2 = args.checkpoints[1]
    chkpt1    = load_checkpoint(filename1)
    chkpt2    = load_checkpoint(filename2)

    ts1 = DavidTimeseries(chkpt1)
    ts2 = DavidTimeseries(chkpt2)

    start_time_1 = ts1.modelparams['inspiral_start_time']
    start_time_2 = ts2.modelparams['inspiral_start_time']

    Number_Of_Orbits = 100
    interest1= (ts1.time < start_time_1) & (ts1.time > start_time_1 - Number_Of_Orbits)
    interest2 = (ts2.time < start_time_2) & (ts2.time > start_time_2 - Number_Of_Orbits)
    Orbits1 = ts1.time[interest1]
    Orbits2 = ts2.time[interest2]

    n = 4096

    if args.Accretion:
        mdot_1 = ts1.mdot1[interest1]+ts1.mdot2[interest1]
        mdot_2 = ts2.mdot1[interest2]+ts2.mdot2[interest2]
        time_1 = ts1.time[interest1]
        time_2 = ts2.time[interest2]
        Mean_Norm_Factor_1 = np.array(np.mean(mdot_1))
        Mean_Norm_Factor_2 = np.array(np.mean(mdot_2))
        mdot_1 = mdot_1 / Mean_Norm_Factor_1
        mdot_2 = mdot_2 / Mean_Norm_Factor_2
        sampling_rate_1 = n / (time_1[-1] - time_1[0])
        sampling_rate_2 = n / (time_2[-1] - time_2[0])
        fq1, p1 = periodogram(mdot_1,fs=sampling_rate_1)
        fq2, p2 = periodogram(mdot_2,fs=sampling_rate_2)
        plt.plot(fq1,p1,label=filename1)
        plt.plot(fq2,p2,label=filename2)
        plt.xlabel(r'$\omega$')
        plt.ylim(0,10)
        plt.legend()
        plt.show()
    if args.Lightcurves:
        Xray1 = ts1.uv[interest1]
        Xray2 = ts2.uv[interest2]
        Norm1 = np.array(np.mean(Xray1))
        Norm2 = np.array(np.mean(Xray2))
        Xray1 = Xray1 / Norm1
        Xray2 = Xray2 / Norm2
        time_1 = ts1.time[interest1]
        time_2 = ts2.time[interest2]
        sampling_rate_1 = n / (time_1[-1] - time_1[0])
        sampling_rate_2 = n / (time_2[-1] - time_2[0])
        fq1, p1 = periodogram(Xray1,fs=sampling_rate_1)
        fq2, p2 = periodogram(Xray2,fs=sampling_rate_2)
        print(p1)
        plt.semilogx(1 / fq1,p1,label=filename1)
        plt.semilogx(1 / fq2,p2,label=filename2)
        #plt.xlabel(r'$\omega$')
        plt.xlabel("P")
        plt.xlim(0,3.0)
        plt.ylim(0,2)
        plt.legend()
        plt.show()
