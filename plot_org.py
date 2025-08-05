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
    parser.add_argument(
        "--radial",
        action="store_true",
        help="radial profile",
    )

    args = parser.parse_args()
    class TorqueCalculation:
        def __init__(self, mesh, masses):
            self.mesh = mesh
            self.masses = masses

        def __call__(self, primitive):
            mesh   = self.mesh
            ni, nj = mesh.shape
            dx     = mesh.dx
            dy     = mesh.dy
            da     = dx * dy
            x      = np.array([mesh.cell_coordinates(i, 0)[0] for i in range(ni)])[:, None]
            y      = np.array([mesh.cell_coordinates(0, j)[1] for j in range(nj)])[None, :]

            x1  = self.masses[0].position_x
            y1  = self.masses[0].position_y
            x2  = self.masses[1].position_x
            y2  = self.masses[1].position_y
            m1  = self.masses[0].mass
            m2  = self.masses[1].mass
            rs1 = self.masses[0].softening_length
            rs2 = self.masses[1].softening_length

            sigma = primitive[:, :, 0]
            delx1 = x - x1
            dely1 = y - y1
            delx2 = x - x2
            dely2 = y - y2

            # forces on the gas
            fx1 = -sigma * da * m1 * delx1 / (delx1**2 + dely1**2 + rs1**2) ** 1.5
            fy1 = -sigma * da * m1 * dely1 / (delx1**2 + dely1**2 + rs1**2) ** 1.5
            fx2 = -sigma * da * m2 * delx2 / (delx2**2 + dely2**2 + rs2**2) ** 1.5
            fy2 = -sigma * da * m2 * dely2 / (delx2**2 + dely2**2 + rs2**2) ** 1.5

            t1 = x * fy1 - y * fx1
            t2 = x * fy2 - y * fx2
            t = t1 + t2
            print("total torque:", t.sum())
            return np.abs(t) ** 0.125 * np.sign(t)
    #Numerical Implementation
    #Needs checking - analytic from solvers?
    class VelocityDivergence:
        def __init__(self, mesh):
            self.mesh = mesh
            
        def __call__(self, primitive):
            mesh = self.mesh
            ni, nj = mesh.shape
            dx     = mesh.dx
            dy     = mesh.dy
            
            vx = primitive[:,:,1]
            vy = primitive[:,:,2]
            
            dvxdx = np.gradient(vx,dx,axis=1)
            dvydy = np.gradient(vy,dy,axis=0)
            divv = dvxdx + dvydy
            
            return np.sign(divv) * np.abs(divv) ** 0.125
            
        


    for filename in args.checkpoints:
        fig, ax     = plt.subplots(figsize=[10, 10])
        chkpt       = load_checkpoint(filename, require_solver="cbdgam_2d")
        CurrentTime = chkpt["time"]/ 2 / np.pi
        mesh        = chkpt["mesh"]
        prim        = chkpt["solution"]
        

        gamma = 5.0/3.0
    
        # Write Luminosity arguments
        if True:
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
        cbar = fig.colorbar(cm)
        cbar.set_label(r'$\Sigma$',fontsize=16)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)

        ax.set_aspect("equal")
        #fig.suptitle(chkpt["time"]/2/np.pi)

        fig.subplots_adjust(
        left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0, wspace=0
        )

        ax.set_xlabel(r'$a_0$',fontsize=16)
        ax.set_ylabel(r'$a_0$',fontsize=16)
        fig.tight_layout()


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
                exit(main_cbdgam_2d())
            # else:
            #     print(f"Unknown solver {chkpt['solver']}")
        
