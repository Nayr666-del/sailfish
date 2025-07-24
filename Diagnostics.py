import sys 
import numpy as np 
import pickle as pk
import matplotlib.pyplot as plt 
import sailfish
import os
import argparse
from sailfish.setup_base import SetupBase
from sailfish.physics.kepler import OrbitalState, PointMass
import matplotlib
# matplotlib.use('Agg')  # Use Agg backend for non-interactive plotting
from cooling import cgs


class FixNumpyCoreUnpickler(pk.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)


def load_checkpoint(filename, require_solver=None):
    with open(filename, "rb") as f:
        chkpt = FixNumpyCoreUnpickler(f).load()
    return chkpt

# def E_from_M(M, e=1.0):
#     f = lambda E: E - e * np.sin(E) - M
#     E = scipy.optimize.root_scalar(f, x0=M, x1=M + 0.1, method='secant').root
#     return E

#current time in code units as well

def to_real_time(user_time,a0,M):
    real_time = user_time * 2 * np.pi * (a0**3)**0.5 * cgs['G'] * cgs['msun'] * M / (cgs['c']**3)
    return real_time
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
        
        
# time in code units, dt in real units
# time * 2*np.pi is the real time
    @property
    def dt(self):#list of dts
        return np.r_[0.0, np.diff(self.time * 2 * np.pi)]
    
    @property
    def mean_anomaly(self):
        return self.time * 2 * np.pi

    #@property
    #def eccentric_anomaly(self):
    #    return np.array([E_from_M(x, e=e) for x, e in zip(self.mean_anomaly, self.eccentricity)])

    @property
    def binary_torque(self):
        return self.torque_g + self.torque_a

    @property
    def binary_delta_j(self):
        return (self.torque_g + self.torque_a) * self.dt

    @property
    def buffer_delta_j(self):
        return self.torque_a * self.dt

    @property
    def total_angular_momentum(self):
    	return self.jdisk + self.binary_delta_j + self.buffer_delta_j # self.gw_delta_j
      
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", type=str, nargs="+")
    parser.add_argument(
        "--Output",
        "-o",
        default=None,
        type=str,
        help="Where to save the output png files",
    )
    parser.add_argument(
        "--Real_Time",
        "-rt",
        default=False,
        action='store_true',
        help="Plot in physical time")
    parser.add_argument(
        "--Disk_Momentum",
        "-jd",
        action='store_true',
        help="whether to plot the total change in momentum timeseries",
    )

    parser.add_argument(
        "--Torque_Components",
        "-t",
        action='store_true',
        help="whether to plot the torque components from the binary",
    )
    parser.add_argument(
        "--Accretion",
        "-a",
        action='store_true',
        help="whether to plot the binary's accretion timeseries",
    )
    parser.add_argument(
        "--Orbital_Elements",
        "-OE",
        action='store_true',
        help="whether to plot the binary's changing orbital elements",
    )
    parser.add_argument(
        "--Power_Components",
        "-p",
        action='store_true',
        help="whether to plot the power exerted on the binary",
    )
    parser.add_argument(
        "--Accreted_Energy",
        "-ae",
        action='store_true',
        help="whether to plot the energy of the gas accreted by the binary",
    )
    parser.add_argument(
        "--Energy",
        "-e",
        action='store_true',
        help="whether to plot the energy emmitted by the disk",
    )
    parser.add_argument(
        "--Lightcurves",
        "-lc",
        action='store_true',
        help="whether to plot the optical and infared luminosities of the disk",
    )
    parser.add_argument(
        "--FloorCount",
        "-fc",
        action='store_true',
        help="whether to plot the number of cells that have reached the floor values",
    )
    parser.add_argument(
        "--Max_temperature",
        "-MT",
        action='store_true',
        help="whether to plot the maximum temperature of the disk",
    )
    args = parser.parse_args()
    

    filename            = args.checkpoints[0]
    chkpt               = load_checkpoint(filename)
    ts                  = DavidTimeseries(chkpt)

    try:
        Primary,Secondary   = ts.pointmasses
        Point_MassPrimary   = PointMass(Primary.mass, Primary.position_x, Primary.position_y, Primary.velocity_x, Primary.velocity_y)
        Point_MassSecondary = PointMass(Secondary.mass,Secondary.position_x,Secondary.position_y,Secondary.velocity_x,Secondary.velocity_y)
        OrbitalEccentricity = OrbitalState(Point_MassPrimary,Point_MassSecondary).eccentricity
    except: # for the case of merged
        Primary             = ts.pointmasses
        OrbitalEccentricity = 0.


    CurrentTime         = ts.currenttime
    Model_Parameters    = ts.modelparams

    Number_of_Orbits    = 100.
    Final_Orbits        = ts.time[ts.time>CurrentTime-Number_of_Orbits] # final 100 orbits
    TimeBins            = np.arange(Final_Orbits[0],Final_Orbits[-1],1)

    hist, edges         = np.histogram(Final_Orbits, bins=int(Number_of_Orbits))
    CumulativeTimeBin   = np.cumsum(hist)
    alpha               = Model_Parameters["alpha"]

    if args.Max_temperature:
        plt.figure()
        plt.plot(ts.time[-len(Final_Orbits):], ts.Max_temp[-len(Final_Orbits):], c = 'black')
        plt.xlabel('time')
        plt.title('Maximum Temperature e = %g'%(np.round(OrbitalEccentricity,3)))
        plt.yscale('log')
        #plt.ylim([1e5,1e7])
        plt.show()
        try:
            savename = os.getcwd() + "/MaxTemperature.%04d.png"%(CurrentTime)
            plt.savefig(savename, dpi=400)
        except:
            plt.show()

    if args.FloorCount:
        plt.figure()
        plt.plot(ts.time[-len(Final_Orbits):], ts.density_floor[-len(Final_Orbits):], c = 'black', label = r'$N_\mathrm{cells}$ at density floor')     
        plt.plot(ts.time[-len(Final_Orbits):], ts.pressure_floor[-len(Final_Orbits):], c = 'black', linestyle ='dashed', label = r'$N_\mathrm{cells}$ at pressure floor')     
        plt.plot(ts.time[-len(Final_Orbits):], ts.uncounted_cells[-len(Final_Orbits):], c = 'red', label = r'$N_\mathrm{cells}$ ignored by lightcurves') 
        plt.axhline(y = 4000000)
        #plt.ylim([0,10])
        
        plt.xlabel('time')
        plt.legend()
        try:
            savename = args.Output + "/FloorCount.%04d.png"%(CurrentTime)
            plt.savefig(savename, dpi=400)
        except:
            plt.show()
        #print("Nu",ts.floor)
    
    if args.Lightcurves:
        StartTime = Model_Parameters['inspiral_start_time']
        MergeTime = Model_Parameters['gw_inspiral_time'] / (2 * np.pi) + StartTime
        if args.Real_Time:
            a0 = Model_Parameters['init_separation_rg']
            M = Model_Parameters['central_mass_msun']
            Final_Orbits = to_real_time(Final_Orbits, a0, M) / 24 / 3600
            StartTime = to_real_time(StartTime, a0, M) / 24 / 3600
            MergeTime = to_real_time(MergeTime, a0, M) / 24 / 3600
            CurrentTime = to_real_time(CurrentTime, a0, M) / 24 / 3600
            Number_of_Orbits = to_real_time(Number_of_Orbits, a0, M) / 24 / 3600
        plt.figure(figsize = (10,3))
        plt.plot(Final_Orbits, ts.infared[-len(Final_Orbits):], c = 'red', label = 'infared luminosity')
        plt.plot(Final_Orbits, ts.optical[-len(Final_Orbits):], c = 'blue', label = 'optical luminosity')
        plt.plot(Final_Orbits, ts.uv[-len(Final_Orbits):],   c = 'purple', label = 'uv')
        plt.plot(Final_Orbits, ts.xray[-len(Final_Orbits):], c = 'green', label = 'xray', linewidth = 0.6) 
        plt.plot(Final_Orbits, ts.bolometric[-len(Final_Orbits):], c = 'black', label = 'bolometric luminosity', linewidth = 0.6)
        plt.legend()
        # plt.plot(ts.time, ts.infared, c = 'red', label = 'infared luminosity')
        # plt.plot(ts.time, ts.optical, c = 'blue', label = 'optical luminosity')
        # plt.plot(ts.time, ts.uv,   c = 'purple', label = 'uv')
        # plt.plot(ts.time, ts.xray, c = 'green', label = 'xray', linewidth = 0.6) 
        # plt.plot(ts.time, ts.bolometric, c = 'black', label = 'bolometric luminosity', linewidth = 0.6)
        #plt.axvline(x = StartTime, linestyle = 'dashed', label ='Inspiral start', c = 'gray')
        plt.axvline(x = MergeTime, linestyle = 'dashed', label ='Merger', c = 'black')
        if args.Real_Time:
            plt.xlabel('time (days)')
        else:
            plt.xlabel('time')
        plt.ylabel('erg/s')
        plt.title('Multiband Lightcurves')
        plt.yscale('log')
        plt.ylim([1e35, 1e48])
        print(ts.xray)
        #plt.xlim([180,220])
        plt.show()
        if args.Real_Time:
            try:
                savename = args.Output + "/Lightcurves_Days.%04d.png"%(CurrentTime)
                plt.savefig(savename, dpi=400, bbox_inches='tight')
            except:
                plt.show()
        else:
            try:
                savename = args.Output + "/Lightcurves.%04d.png"%(CurrentTime)
                plt.savefig(savename, dpi=400, bbox_inches='tight')
            except:
                plt.show()

    if args.Disk_Momentum:
        plt.figure()
        plt.plot(ts.time, ts.total_angular_momentum, c = 'black')
        plt.xlabel('time')
        plt.title('Total Angular Momentum e = %g Progradeay'%(np.round(OrbitalEccentricity,3)))
        plt.show()
        try:
            savename = os.getcwd() + "/TotalAngularMomentum.%04d.png"%(CurrentTime)
            plt.savefig(savename, dpi=400)
        except:
            plt.show()


    if args.Torque_Components:
        #InnerClipped_Torque = ts.innertorque[-len(Final_Orbits):] / M_dot_0
        #OuterClipped_Torque = ts.outertorque[-len(Final_Orbits):] / M_dot_0
        Normalised_Torque_g = ts.torque_g[-len(Final_Orbits):] #/ M_dot_0
        Normalised_Torque_a = ts.torque_a[-len(Final_Orbits):] #/ M_dot_0


        plt.figure()
        plt.xlabel('time')
        if Model_Parameters['retrograde']:
            plt.title(r'Torque Retrograde $\alpha = %g$'%(chkpt['model_parameters']['alpha']))
        else:
            plt.title(r'Torque Prograde $\alpha = %g$'%(chkpt['model_parameters']['alpha']))
        
        MeanTorque_g = [np.mean(Normalised_Torque_g[CumulativeTimeBin[i-1]:CumulativeTimeBin[i]]) for i in range(1,len(TimeBins))]
        MeanTorque_a = [np.mean(Normalised_Torque_a[CumulativeTimeBin[i-1]:CumulativeTimeBin[i]]) for i in range(1,len(TimeBins))]

        
        #plt.plot(Final_Orbits,Normalised_Torque_g, c = 'blue', linewidth = 0.1)
        plt.plot(TimeBins[1:],MeanTorque_g,linewidth = 0.5, label = 'Binned Torque Mean Gravitational', c = 'black')
        plt.plot(TimeBins[1:],MeanTorque_a,linewidth = 0.5, label = 'Binned Torque Mean Accretion',linestyle = 'dashed', c = 'black')
        
        #plt.plot(ts.time, ts.jdisk)
        #plt.xlim([CurrentTime-Number_of_Orbits,CurrentTime])


        #plt.axvline(x = 1000., linestyle = 'dashed', label ='Inspiral start', c = 'gray')
        plt.legend(loc = 'upper right')
        #plt.ylim([-2.5,5])
        plt.ylabel(r'$\tau/\dot{M}_0$')
        plt.show()
        try:
            savename = args.Output +  "/MeanTorque.%04d_alpha%g.png"%(CurrentTime,chkpt['model_parameters']['alpha'])
            plt.savefig(savename, dpi=400)
        except:
            plt.show()


        #print('Torque Mean at t=1000 is',MeanTorque_g[0]+MeanTorque_a[0])




    if args.Power_Components: 
        Normalised_Power    = (ts.power_g1[-len(Final_Orbits):]+ts.power_g2[-len(Final_Orbits):]) / M_dot_0
        InnerClipped_Power  = (ts.innerpower_1[-len(Final_Orbits):]+ts.innerpower_2[-len(Final_Orbits):]) / M_dot_0
        OuterClipped_Power  = (ts.outerpower_1[-len(Final_Orbits):]+ts.outerpower_2[-len(Final_Orbits):]) / M_dot_0
        
        plt.figure()
        plt.xlabel('time')
        if Model_Parameters['retrograde']:
            plt.title(r'Power Retrograde $\nu = %g$'%(viscosity))
        else:
            plt.title(r'Power Prograde $\nu = %g$'%(viscosity))

        MeanPower = [np.mean(Normalised_Power[CumulativeTimeBin[i-1]:CumulativeTimeBin[i]]) for i in range(1,len(TimeBins))]
        
        plt.xlim([CurrentTime-Number_of_Orbits,CurrentTime])
        plt.plot(Final_Orbits,Normalised_Power, c = 'Purple', label = 'Power', linewidth = 0.1,)
        plt.plot(TimeBins[1:],MeanPower,linewidth = 0.5, label = 'Binned Means', c = 'black')
        plt.axvline(x = 1000., linestyle = 'dashed', label ='Inspiral start', c = 'gray')
        plt.legend(loc = 'upper right')
        #plt.ylim([-10,10])
        plt.ylabel(r'$\mathcal{P}/\dot{M}_0$')
        try:
            savename = args.Output +  "/MeanPower.%04d_nu%g.png"%(CurrentTime,viscosity)
            plt.savefig(savename, dpi=400)
        except:
            plt.show()




    if args.Accretion:
        try:
            StartTime = Model_Parameters['inspiral_start_time']
            MergeTime = Model_Parameters['gw_inspiral_time'] / (2 * np.pi) + StartTime
        except KeyError:
            pass
        if CurrentTime < Number_of_Orbits:
            Mean_Norm_Factor = np.array(np.mean(ts.mdot1+ts.mdot2))
        else: # Using the last 100 orbits before the final as a normalization mean. Else use the entire.
            Mean_Norm_Factor = np.array(np.mean(ts.mdot1[-len(Final_Orbits)-100:-len(Final_Orbits)]+ts.mdot2[-len(Final_Orbits)-100:-len(Final_Orbits)]))
        plt.figure()
        if args.Real_Time:
            a0 = Model_Parameters['init_separation_rg']
            M = Model_Parameters['central_mass_msun']
            Final_Orbits = to_real_time(Final_Orbits, a0, M) / 24 / 3600
            StartTime = to_real_time(StartTime, a0, M) / 24 / 3600
            MergeTime = to_real_time(MergeTime, a0, M) / 24 / 3600
            CurrentTime = to_real_time(CurrentTime, a0, M) / 24 / 3600
            Number_of_Orbits = to_real_time(Number_of_Orbits, a0, M) / 24 / 3600
        plt.plot(Final_Orbits,(ts.mdot1[-len(Final_Orbits):]+ts.mdot2[-len(Final_Orbits):])/Mean_Norm_Factor,label='mdot',linewidth = 0.5, c = 'red')
        # plt.plot(ts.time,(ts.mdot1+ts.mdot2)/Mean_Norm_Factor,label='mdot',linewidth = 0.5, c = 'red')

        if args.Real_Time:
            plt.xlabel('Time (days)')
        else:
            plt.xlabel('Time [P]')
        plt.ylabel(r'$\dot{M}/\langle\dot{M}_0\rangle$')
        plt.title(r'Accretion Rate, $\alpha=%g$'%(alpha))
        try:
            plt.axvline(x = StartTime, linestyle = 'dashed', label ='Inspiral start', c = 'gray')
            plt.axvline(x = MergeTime, linestyle = 'dashed', label ='Merger', c = 'black')
        except:
            pass
       
        #plt.ylim([0,2])
        if CurrentTime < Number_of_Orbits:
            plt.xlim([0,CurrentTime])
        else:
            plt.xlim([CurrentTime-Number_of_Orbits,CurrentTime])
            pass
        #AccretionRate = (ts.mdot1[-len(Final_Orbits):]+ts.mdot2[-len(Final_Orbits):])#/M_dot_0
        #MeanAccretion = np.array([np.mean(AccretionRate[CumulativeTimeBin[i-1]:CumulativeTimeBin[i]]) for i in range(1,len(TimeBins))])/Mean_Norm_Factor
        #plt.plot(Final_Orbits,MeanAccretion[-len(Final_Orbits):],linewidth = 0.5, label = 'Binned Means', c = 'black')
        plt.legend(loc = 'upper right')
        plt.yscale('log')
        if args.Output is None:
            plt.show()
        else:
            if args.Real_Time:
                savename = args.Output +  "/AccretionRate_Days.%04d_alpha%g.png"%(CurrentTime,alpha)
                plt.savefig(savename, dpi=400)
            else:
                savename = args.Output +  "/AccretionRate.%04d_alpha%g.png"%(CurrentTime,alpha)
                plt.savefig(savename, dpi=400)


    if args.Orbital_Elements:
        plt.figure()
        plt.plot(ts.time,ts.semimajor_axis, label = 'SemiMajor Axis')
        plt.plot(ts.time,ts.eccentricity, label = 'Eccentricity')
        plt.title(r'Orbital Elements $e_0 =$%g Retrograde'%(np.round(OrbitalEccentricity,3)))
        plt.xlabel('Time')
        plt.ylabel('Orbital Elements')
        plt.legend()
        try:
            savename = args.Output +  "/OrbitalElements.%04d.png"%(CurrentTime)
            plt.savefig(savename, dpi=400)
        except:
            plt.show()






