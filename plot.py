import argparse
import pickle as pk
import sys
from sailfish.physics.kepler import OrbitalState
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('Agg')
# import sailfish
from cooling import OpticalEmission, InfaredEmission, XrayEmission, UVEmission
class FixNumpyCoreUnpickler(pk.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)



def load_checkpoint(filename, require_solver=None):
    with open(filename, "rb") as f:
        chkpt = FixNumpyCoreUnpickler(f).load()
    return chkpt

def PrecomputeLum(Band,dx,length_scale,pc):
    logT_low  = 0
    logT_high = 10

    Temperature_Range    = np.logspace(logT_low,logT_high,int(1e6)) 
    Log_Temperature_Diff = np.diff(np.log10(Temperature_Range))[0]
    
    if Band == 'Optical':
        emission = OpticalEmission(Temperature_Range,dx * length_scale * pc)
    elif Band == 'Infared':
        emission = InfaredEmission(Temperature_Range,dx * length_scale * pc)
    elif Band == 'Xray':
        emission = XrayEmission(Temperature_Range, dx * length_scale * pc)
    elif Band == 'UV':
        emission = UVEmission(Temperature_Range, dx * length_scale * pc)
    elif Band == 'Bol':
        return 'Bol'
    else:
        emission = None
        print('Optical, Infared, UV or Xray, or Bol')
    return emission, Temperature_Range, Log_Temperature_Diff

    
def InterpolateLum(Band,dx,length_scale,pc,T): #T is an array of maps
    Precomputed, Range, Diff = PrecomputeLum(Band,dx,length_scale,pc)
    lowlogT = np.log10(Range[0])
    Tarray = np.maximum(T,10**lowlogT)
    
    Progress = (np.log10(Tarray) - lowlogT) / Diff 
    N0 = np.floor(Progress).astype(int)
    Bracket_N0_N1 = Progress - N0
    
    Emission_N0 = Precomputed[N0]
    Emission_N1 = Precomputed[N0 + 1]
    
    Interpolated_Emission = Emission_N0 + Bracket_N0_N1 * (Emission_N1 - Emission_N0)
    return Interpolated_Emission
    
def ZoomIn2D(chkpt, size=None):
    mesh = chkpt["mesh"]
    prim = chkpt["solution"]

    primary, secondary = chkpt["point_masses"]
    xprim, yprim = primary.position_x, primary.position_y
    xsec, ysec = secondary.position_x, secondary.position_y
    xcenter = (xprim + xsec) / 2
    ycenter = (yprim + ysec) / 2
    x0, y0 = mesh.x0, mesh.y0

    x_index_0 = int((xcenter - size - x0) / mesh.dx)
    y_index_0 = int((ycenter - size - y0) / mesh.dy)
    x_index_1 = int((xcenter + size - x0) / mesh.dx)
    y_index_1 = int((ycenter + size - y0) / mesh.dy)

    di = (x_index_0, x_index_1)
    dj = (y_index_0, y_index_1)
    mesh_zoom = mesh.sub_mesh(di, dj)
    prim_zoom = prim[di[0]:di[1], dj[0]:dj[1], :]
    new_resolution = mesh_zoom.ni


    chkpt["solution"] = prim_zoom
    chkpt["mesh"] = mesh_zoom._replace(ni=mesh_zoom.ni, nj=mesh_zoom.nj)
    chkpt["model_parameters"]["domain_radius"] = size
    chkpt["model_parameters"]["sink_radius"] = chkpt["model_parameters"]["sink_radius"]
    chkpt["model_parameters"]["softening_length"] = chkpt["model_parameters"]["softening_length"]
    return chkpt   

        
    


def main_cbdgam_2d():
    import numpy as np

    fields = {
        "sigma": lambda p: p[:, :, 0],
        "vx": lambda p: p[:, :, 1],
        "vy": lambda p: p[:, :, 2],
        "pre": lambda p: p[:, :, 3],
    }



    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", type=str, nargs="+")
    parser.add_argument(
        "--field",
        "-f",
        type=str,
        default="sigma",
        help="which field to plot",
    )
    parser.add_argument("--poly", type=int, nargs=2, default=None)
    parser.add_argument(
        "--log",
        "-l",
        default=False,
        action="store_true",
        help="use log scaling",
    )
    parser.add_argument(
        "--SED",
        default=False,
        action="store_true",
        help="plot spectrum",
    )
    parser.add_argument(
        "--vmap",
        default=False,
        action="store_true",
        help="plot gas velocities",
    )
    parser.add_argument(
        "--vmin",
        default=None,
        type=float,
        help="minimum value for colormap",
    )
    parser.add_argument(
        "--vmax",
        default=None,
        type=float,
        help="maximum value for colormap",
    )
    parser.add_argument(
        "--radius",
        default=None,
        type=float,
        help="plot the domain out to this radius",
    )
    parser.add_argument(
        "--Outputs",
        "-o",
        default=None,
        type=str,
        help="Where to save the output png files",
    )
    parser.add_argument(
        "--print_model_parameters",
        "-params",
        action="store_true",
        help="plot the parameters used for making this checkpoint",
    )
    parser.add_argument(
        "--plot_sink",
        action="store_true",
        help="plot the sink properties",
    )
    parser.add_argument(
        "--cmap",
        default="magma",
        help="colormap name",
    )
    parser.add_argument(
        "--centered",
        "-c",
        default=None,
        type=float,
        help="center the plot with radius",
    )

    args = parser.parse_args()

    for filename in args.checkpoints:
        fig, ax     = plt.subplots(figsize=[10, 10])
        chkpt       = load_checkpoint(filename, require_solver="cbdgam_2d")
        if args.centered is not None:
            chkpt = ZoomIn2D(chkpt, size=args.centered)
        CurrentTime = chkpt["time"]/ 2 / np.pi
        mesh        = chkpt["mesh"]
        prim        = chkpt["solution"]

        
        import cooling
        from cooling import gamma_law_index, EffectiveTemperature, cgs

        gamma = gamma_law_index(chkpt['model_parameters']['beta'], chkpt['model_parameters']['gamma_law_index_gas'])
        try:
            length_scale_pc = chkpt['model_parameters']['length_scale_pc']
        except KeyError as e:
            r_g             = cgs['G'] * chkpt['model_parameters']['central_mass_msun'] * cgs['msun'] / cgs['c'] / cgs['c']
            length_scale_pc = r_g * chkpt['model_parameters']['init_separation_rg'] / cgs['pc']

        SS73 = cooling.ShakuraSunyaevDisk(
            central_mass_msun = chkpt['model_parameters']['central_mass_msun'], 
            length_scale_pc   = length_scale_pc,
            mach_number_a     = chkpt['model_parameters']['mach_number_a'],
            alpha             = chkpt['model_parameters']['alpha'],
            gamma             = gamma
            )
        # Write Luminosity arguments
        if args.field == 't':
            Sigma    = fields["sigma"](prim)
            Pressure = fields["pre"](prim)

            kb_code    = cgs['kb'] / (SS73._mass * SS73._length**2 / SS73._time**2)
            mp_code    = cgs['mp'] / (SS73._mass)
            kappa_code = cgs['kappa'] / (SS73._length**2 / SS73._mass)

            Midplane_T    = ((Pressure / Sigma) * (mp_code / kb_code))
            optical_depth = Sigma * kappa_code
            mask_values   = optical_depth >= 1.0 * SS73._eddington_fraction / chkpt['model_parameters']['target_accretion_rate'] ####FIX THIS MASK
            
            Teff          = EffectiveTemperature(optical_depth, Midplane_T)
            EddingtonFrac = SS73._eddington_fraction
            RescaledTemp  = Teff * (chkpt['model_parameters']['target_accretion_rate']/EddingtonFrac) ** 0.25
            print('Remapping factor', chkpt['model_parameters']['target_accretion_rate']/EddingtonFrac)
            f             = (RescaledTemp * mask_values).T
            f[f<=1] = 1
        elif args.field == "Xray":
            Sigma = fields['sigma'](prim)
            Pressure = fields['pre'](prim)
            
            kb_code = cgs['kb'] / (SS73._mass * SS73._length**2 / SS73._time**2)
            mp_code = cgs['mp'] / (SS73._mass)
            kappa_code = cgs['kappa'] / (SS73._length**2 / SS73._mass)
            
            Midplane_T = ((Pressure / Sigma) * (mp_code / kb_code))
            optical_depth = Sigma * kappa_code
            mask_values   = optical_depth >= 1.0 * SS73._eddington_fraction / chkpt['model_parameters']['target_accretion_rate'] ####FIX THIS MASK
            
            Teff          = EffectiveTemperature(optical_depth, Midplane_T)
            EddingtonFrac = SS73._eddington_fraction
            RescaledTemp  = Teff * (chkpt['model_parameters']['target_accretion_rate']/EddingtonFrac) ** 0.25 * mask_values
            Lum = InterpolateLum('Xray', mesh.dx, length_scale_pc,cgs['pc'],RescaledTemp)
            Lum[Lum<=1] = 1
            f = (Lum).T
        elif args.field == "UV":
            Sigma = fields['sigma'](prim)
            Pressure = fields['pre'](prim)
            
            kb_code = cgs['kb'] / (SS73._mass * SS73._length**2 / SS73._time**2)
            mp_code = cgs['mp'] / (SS73._mass)
            kappa_code = cgs['kappa'] / (SS73._length**2 / SS73._mass)
            
            Midplane_T = ((Pressure / Sigma) * (mp_code / kb_code))
            optical_depth = Sigma * kappa_code
            mask_values   = optical_depth >= 1.0 * SS73._eddington_fraction / chkpt['model_parameters']['target_accretion_rate'] ####FIX THIS MASK
            
            Teff          = EffectiveTemperature(optical_depth, Midplane_T)
            EddingtonFrac = SS73._eddington_fraction
            RescaledTemp  = Teff * (chkpt['model_parameters']['target_accretion_rate']/EddingtonFrac) ** 0.25 * mask_values
            Lum = InterpolateLum('UV', mesh.dx, length_scale_pc,cgs['pc'],RescaledTemp)
            Lum[Lum<=1] = 1
            f = (Lum).T
        elif args.field == "Optical":
            Sigma = fields['sigma'](prim)
            Pressure = fields['pre'](prim)
            
            kb_code = cgs['kb'] / (SS73._mass * SS73._length**2 / SS73._time**2)
            mp_code = cgs['mp'] / (SS73._mass)
            kappa_code = cgs['kappa'] / (SS73._length**2 / SS73._mass)
            
            Midplane_T = ((Pressure / Sigma) * (mp_code / kb_code))
            optical_depth = Sigma * kappa_code
            mask_values   = optical_depth >= 1.0 * SS73._eddington_fraction / chkpt['model_parameters']['target_accretion_rate'] ####FIX THIS MASK
            
            Teff          = EffectiveTemperature(optical_depth, Midplane_T)
            EddingtonFrac = SS73._eddington_fraction
            RescaledTemp  = Teff * (chkpt['model_parameters']['target_accretion_rate']/EddingtonFrac) ** 0.25 * mask_values
            Lum = InterpolateLum('Optical', mesh.dx, length_scale_pc,cgs['pc'],RescaledTemp)
            Lum[Lum<=1] = 1
            f = (Lum).T
        elif args.field == "Infared":
            Sigma = fields['sigma'](prim)
            Pressure = fields['pre'](prim)
            
            kb_code = cgs['kb'] / (SS73._mass * SS73._length**2 / SS73._time**2)
            mp_code = cgs['mp'] / (SS73._mass)
            kappa_code = cgs['kappa'] / (SS73._length**2 / SS73._mass)
            
            Midplane_T = ((Pressure / Sigma) * (mp_code / kb_code))
            optical_depth = Sigma * kappa_code
            mask_values   = optical_depth >= 1.0 * SS73._eddington_fraction / chkpt['model_parameters']['target_accretion_rate'] ####FIX THIS MASK
            
            Teff          = EffectiveTemperature(optical_depth, Midplane_T)
            EddingtonFrac = SS73._eddington_fraction
            RescaledTemp  = Teff * (chkpt['model_parameters']['target_accretion_rate']/EddingtonFrac) ** 0.25 * mask_values
            Lum = InterpolateLum('Infared', mesh.dx, length_scale_pc,cgs['pc'],RescaledTemp)
            Lum[Lum<=1] = 1
            f = (Lum).T
        elif args.field == 'mach':# Why? shouldn't it be vx**2 + vy**2?
             Sigma    = fields["sigma"](prim)
             Pressure = fields["pre"](prim)
 
             cs     = (gamma * Pressure / Sigma)**0.5
             vx = fields['vx'](prim)
             vy = fields['vy'](prim)
             
             v = (vx**2 + vy**2 ) **0.5
             
             Mach = v / cs
             f = Mach.T
             # ni, nj = mesh.shape
             # xspace = np.linspace(mesh.x0, mesh.x1,ni)
             # yspace = np.linspace(mesh.y0, mesh.y1,nj)
             # X, Y   = np.meshgrid(xspace, yspace)
             # primary, secondary = chkpt['point_masses']
             # xprim, yprim = primary.position_x, primary.position_y
             # xsec, ysec   = secondary.position_x, secondary.position_y

             # R_1     = np.sqrt((X-xprim)**2 + (Y-yprim)**2)
             # R_2     = np.sqrt((X-xsec)**2  + (Y-ysec)**2)
             # Omega_1 = np.sqrt(0.5 / (R_1**3 + 1e-12)) #Assuming a Keplerian disk
             # Omega_2 = np.sqrt(0.5 / (R_2**3 + 1e-12))  #Assuming a Keplerian disk
             
             # ROmega  = np.sqrt((R_1 * Omega_1)**2 + (R_2 * Omega_2)**2) #Assuming a Keplerian disk
 
             # Mach    = ROmega / cs
             # f       = Mach.T

        elif args.field == 'h':
            Sigma    = fields["sigma"](prim)
            Pressure = fields["pre"](prim)
 
            cs     = (gamma * Pressure / Sigma)**0.5
            
            ni, nj = mesh.shape
            xspace = np.linspace(mesh.x0, mesh.x1,ni)
            yspace = np.linspace(mesh.y0, mesh.y1,nj)
            X, Y   = np.meshgrid(xspace, yspace)
            primary, secondary = chkpt['point_masses']
            xprim, yprim = primary.position_x, primary.position_y
            xsec, ysec   = secondary.position_x, secondary.position_y
            
            R_1     = np.sqrt((X-xprim)**2 + (Y-yprim)**2)
            R_2     = np.sqrt((X-xsec)**2  + (Y-ysec)**2)
            Omega_1 = np.sqrt(0.5 / (R_1**3 + 1e-12)) #Assuming a Keplerian disk
            Omega_2 = np.sqrt(0.5 / (R_2**3 + 1e-12))  #Assuming a Keplerian disk
               
            ROmega  = np.sqrt((R_1 * Omega_1)**2 + (R_2 * Omega_2)**2) #Assuming a Keplerian disk
             
            h    = cs / ROmega
            f       = h.T
        elif args.field == 'H':
            Sigma    = fields["sigma"](prim)
            Pressure = fields["pre"](prim)
 
            cs     = (gamma * Pressure / Sigma)**0.5
            
            ni, nj = mesh.shape
            xspace = np.linspace(mesh.x0, mesh.x1,ni)
            yspace = np.linspace(mesh.y0, mesh.y1,nj)
            X, Y   = np.meshgrid(xspace, yspace)
            primary, secondary = chkpt['point_masses']
            xprim, yprim = primary.position_x, primary.position_y
            xsec, ysec   = secondary.position_x, secondary.position_y
            
            R_1     = np.sqrt((X-xprim)**2 + (Y-yprim)**2)
            R_2     = np.sqrt((X-xsec)**2  + (Y-ysec)**2)
            Omega_1 = np.sqrt(0.5 / (R_1**3 + 1e-12)) #Assuming a Keplerian disk
            Omega_2 = np.sqrt(0.5 / (R_2**3 + 1e-12))  #Assuming a Keplerian disk
               
            Omega  = np.sqrt((Omega_1)**2 + (Omega_2)**2) #Assuming a Keplerian disk
             
            H    = cs / Omega
            f       = H.T
        elif args.field == 'tau':# Might need fixing
            Sigma    = fields["sigma"](prim) 
            #Pressure = fields["pre"](prim)

            kb_code    = cgs['kb'] / (SS73._mass * SS73._length**2 / SS73._time**2)
            mp_code    = cgs['mp'] / (SS73._mass)
            kappa_code = cgs['kappa'] / (SS73._length**2 / SS73._mass)

            #Midplane_T    = ((Pressure / Sigma) * (mp_code / kb_code)).T
            #optical_depth = Sigma * kappa_code
            #Teff          = EffectiveTemperature(optical_depth, Midplane_T)
            #EddingtonFrac = SS73._eddington_fraction
            #RescaledTemp  = Teff * (10/EddingtonFrac) ** 0.25

            f         = Sigma * kappa_code * 1.0 / (10 / SS73._eddington_fraction)

            print('Minimum Density', 10 / kappa_code)
            #Tmid = 10 **-2 * mp_code/kb_code
            #kappa_code = cgs['kappa'] / (SS73._length**2 / SS73._mass)
            #Teff = cooling.EffectiveTemperature(1e-10, kappa_code, Tmid)
            #EmittingTemp = Teff * (10/SS73._eddington_fraction)** 0.25
            #print('Emitting Temp',EmittingTemp)
            #print('Optical depth', kappa_code * 1e-10)
        
        elif args.field == 'div-v':
            vx = fields['vx'](prim)
            vy = fields['vy'](prim)
            
            dx = mesh.dx
            dy = mesh.dy
            
            dvxdx = np.gradient(vx,dx,axis=1)
            dvydy = np.gradient(vy,dy,axis=0)
            
            divv = dvxdx + dvydy
            
            f = np.sign(divv) * (np.abs(divv))**0.125
            f = f.T
            # vx = fields['vx'](prim)
            # vy = fields['vy'](prim)
            
            # ni, nj = mesh.shape
            # xspace = np.linspace(mesh.x0, mesh.x1,ni)
            # yspace = np.linspace(mesh.y0, mesh.y1,nj)
            # Y, X   = np.meshgrid(xspace, yspace)
            
            # R = np.sqrt(X**2 + Y**2)
            # ###
        elif args.field == 'vr':
            vx = fields['vx'](prim)
            vy = fields['vy'](prim)
            ni, nj = mesh.shape
            xspace = np.linspace(mesh.x0, mesh.x1,ni)
            yspace = np.linspace(mesh.y0, mesh.y1,nj)
            primary, secondary = chkpt['point_masses']
            xprim, yprim = primary.position_x, primary.position_y
            Y, X   = np.meshgrid(xspace, yspace)
            # softening = chkpt['model_parameters']['softening_length']
            R = np.maximum(np.sqrt(X * X + Y * Y), 1e-12)
            x_hat = X / R
            y_hat = Y / R
            
            vr = x_hat * vx + y_hat * vy
            # phi_x_hat = - Y / np.maximum(R,1e-12)
            # phi_y_hat = + X / np.maximum(R,1e-12)
            # vx0 = np.sqrt(1.0 / R) * phi_x_hat
            # vy0 = np.sqrt(1.0 / R) * phi_y_hat
            
            # dvr = (vx - vx0) * X / np.maximum(R,1e-12) + (vy - vy0) * Y / np.maximum(R,1e-12)
            
            f = vr.T
            
            
             
        elif args.field == 's':
            Sigma = fields['sigma'](prim)
            Pressure = fields['pre'](prim)
            
            f = (Pressure / Sigma ** gamma).T
        else:
            f = fields[args.field](prim).T

        if args.log:
            f = np.log10(f)

        extent = mesh.x0, mesh.x1, mesh.y0, mesh.y1


        cm = ax.imshow(
            f,
            origin="lower",
            vmin=args.vmin,
            vmax=args.vmax,
            cmap=args.cmap,
            extent=extent,
        )
        fig.colorbar(cm)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)

        ax.set_aspect("equal")
        fig.suptitle(chkpt["time"]/2/np.pi)

        fig.subplots_adjust(
        left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0, wspace=0
        )

    if args.SED:
        kb_code    = cgs['kb'] / (SS73._mass * SS73._length**2 / SS73._time**2)
        mp_code    = cgs['mp'] / (SS73._mass)
        kappa_code = cgs['kappa'] / (SS73._length**2 / SS73._mass)

        Sigma           = fields["sigma"](prim) 
        Pressure        = fields["pre"](prim)
        optical_depth   = Sigma * kappa_code
        Remapping_Value = chkpt['model_parameters']['target_accretion_rate'] / SS73._eddington_fraction
        mask_values     = optical_depth >= (10.0 / Remapping_Value)
        T               = np.maximum((Pressure / Sigma) * (mp_code / kb_code), 1) * mask_values

        Teff                  = EffectiveTemperature(optical_depth, T)
        RescaledTemp          = Teff * Remapping_Value ** 0.25
        mask_values           = optical_depth >= (10.0 / Remapping_Value)
        
        ev            = 1.6e-12
        E_low         = 1e-1 * ev
        E_high        = 5e3 * ev
        freq_low      = E_low  / cgs['h']
        freq_high     = E_high / cgs['h']
        E_Xray_low    = cgs['h'] * cgs['c'] / (1e-6) / 1000/ ev

        Ev_array      = np.logspace(np.log10(E_low/1000/ev), np.log10(E_high/1000/ev), 100)  # Energy range in eV
        freq_space    = np.logspace(np.log10(freq_low), np.log10(freq_high), len(Ev_array))  # Frequency range in Hz
        ni, nj        = np.shape(RescaledTemp)
        dx            = mesh.dx

        Cell_Spectra  = np.array([cooling.PlanckSpectrum(freq, RescaledTemp, length_scale_pc * cgs['pc'] * dx) for freq in freq_space])
        Spectrum      = np.sum(np.sum(Cell_Spectra, axis=1), axis=1) 
        integral      = np.trapz(Spectrum, np.log(freq_space)) 

        plt.figure(figsize=(4, 3))
        plt.plot(Ev_array, Spectrum, label = 'SED')
        plt.xscale('log')
        plt.yscale('log') 
        plt.axvline(x = E_Xray_low, linestyle='dashed', c = 'red', label = 'X-ray low frequency')        
        plt.axhline(y = integral, linestyle='dashed', c = 'black', label = 'Total Luminosity')
        plt.legend()
        plt.xlabel(r'$h\nu~[\mathrm{kev}]$')
        plt.ylabel(r'$2\pi\nu B_\nu(T)~[\mathrm{erg/s}]$')
        plt.title('Spectral Energy Distribution at t = %g'%(chkpt["time"]/ 2 / np.pi))
        plt.ylim([1e35, 2 * integral])
        plt.show()
        #plt.savefig(args.Outputs + "/SED_%g.png"%(chkpt["time"]/ 2 / np.pi), dpi=300, bbox_inches='tight')

    if args.vmap:
            Number_of_Vectors = 15
            ni, nj            = mesh.shape
            x                 = np.array([mesh.cell_coordinates(i, 0)[0] for i in range(ni)])[:, None]
            y                 = np.array([mesh.cell_coordinates(0, j)[1] for j in range(nj)])[None, :]
            Vx                = chkpt['solution'][:,:,1].T
            Vy                = chkpt['solution'][:,:,2].T

            try:
                rescaled_x = [ix for ix in x if np.abs(ix) < args.radius]
                xmin, xmax = np.where(x == np.min(rescaled_x))[0][0], np.where(x == np.max(rescaled_x))[0][0]
            except:
                rescaled_x = x
                xmin, xmax = 0,len(x)-1

            Sampling   = (xmax-xmin)//Number_of_Vectors
            X, Y       = np.meshgrid(x[xmin:xmax:Sampling, :], y[:, xmin:xmax:Sampling])
            Vx_sampled = Vx[xmin:xmax:Sampling, xmin:xmax:Sampling] 
            Vy_sampled = Vy[xmin:xmax:Sampling, xmin:xmax:Sampling]# - 0.5
            print(Vx_sampled)
            #print(np.shape(y[:, xmin:xmax:Sampling]))
            #print(np.shape(x[xmin:xmax:Sampling, :]))
            plt.quiver(
                X, Y,
                Vx_sampled, Vy_sampled,
                width=0.0025, angles='xy', scale_units='xy', scale=20, color = 'darkgrey', headwidth=4
            )



    if args.radius is not None:
            ax.set_xlim(-args.radius, args.radius)
            ax.set_ylim(-args.radius, args.radius)

    if args.print_model_parameters:
            print('Iteration Number.........',chkpt['iteration'])
            print('Timestep_dt..............',chkpt['timestep_dt'])
            print('cfl_number...............',chkpt['cfl_number'])
            print('Solver options...........',chkpt['solver_options'])
            print('Event states.............',chkpt['event_states'])

            print('------------------Driver------------------')
            print(chkpt['driver'])
            print('-------------Model Parameters-------------')
            print(chkpt["model_parameters"])
            print('---------------Point Masses---------------')
            print(chkpt["point_masses"])
            print('------------------------------------------')

    if args.plot_sink:
        from sailfish.physics.kepler import OrbitalState, PointMass
        from matplotlib.patches import Circle

        primary, secondary = chkpt['point_masses']
        Primary   = PointMass(primary.mass  , primary.position_x  , primary.position_y  , primary.velocity_x  , primary.velocity_y)
        Secondary = PointMass(secondary.mass, secondary.position_x, secondary.position_y, secondary.velocity_x, secondary.velocity_y)

        orbital_state = OrbitalState(Primary, Secondary)

        ax.scatter(primary.position_x, primary.position_y, marker = '+', s = 40, c = 'white', label = 'Point Mass')
        ax.scatter(secondary.position_x, secondary.position_y, marker = '+', s = 40, c = 'white')

        
        primarycenter   = (primary.position_x, primary.position_y)
        secondarycenter = (secondary.position_x, secondary.position_y)
        radius          = primary.sink_radius         # Radius of the circle
        print(primary.sink_radius)
        primarysink   = Circle(primarycenter, radius, color='grey', fill=True, alpha=0.8)
        secondarysink = Circle(secondarycenter, radius, color='grey', fill=True, alpha=0.8)
        ax.add_patch(primarysink)
        ax.add_patch(secondarysink)

        def Position(t, a, e):
            return 0.5 * a * np.cos(t) - 0.5 * a * e, 0.5 * a * np.sqrt(1 - e**2) * np.sin(t)
        
        eccentr = chkpt['timeseries'][-1][ 2] 
        semimaj = chkpt['timeseries'][-1][ 1] 
        #eccentr = chkpt['timeseries'].eccentricity[-1]
        Orbital_Path = np.array([Position(t, semimaj, eccentr) for t in np.linspace(0,2*np.pi,1000)])
        # USE  TIMESERIES DATA TO GET A AND E AND USE THIS TO PLOT
        plt.plot( Orbital_Path[:,0], Orbital_Path[:,1], linestyle = 'dashed', c = 'grey')
        plt.plot(-Orbital_Path[:,0], Orbital_Path[:,1], linestyle = 'dashed', c = 'grey')

    if args.Outputs is None:
        plt.tight_layout()
        plt.show()
    else:
        pngname     = args.Outputs + f"/{args.field}_Map-{int(CurrentTime * 1000):08d}.png"
        
        fig.savefig(pngname, dpi=400, bbox_inches='tight')


text_width   = 7.1
column_width = text_width / 2.
def configure_matplotlib():
    plt.rc('xtick' , labelsize=8)
    plt.rc('ytick' , labelsize=8)
    plt.rc('axes'  , labelsize=8)
    plt.rc('legend', fontsize=8)
    plt.rc('font', family='DejaVu Sans', size=8)
    plt.rc('text')
configure_matplotlib()

if __name__ == "__main__":
    for arg in sys.argv:
        if arg.endswith(".pk"):
            chkpt = load_checkpoint(arg)
            
            import numpy as np
            print('Time',chkpt['time']/2/np.pi)
            # print(chkpt.keys())
            # print(chkpt['mesh'])
            
            if chkpt["solver"] == "cbdgam_2d":
                print("plotting for cbdgam_2d solver")
                exit(main_cbdgam_2d())
            else:
                print(f"Unknown solver {chkpt['solver']}")
        
