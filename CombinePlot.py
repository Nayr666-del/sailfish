import argparse
import pickle as pk
import sys
from sailfish.physics.kepler import OrbitalState
import matplotlib.pyplot as plt
class FixNumpyCoreUnpickler(pk.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)
    
def load_checkpoint(filename, require_solver=None):
    with open(filename, "rb") as f:
        chkpt = FixNumpyCoreUnpickler(f).load()
    return chkpt

def main_cbdgam_2d():
    import numpy as np

    fields = {
        "sigma": lambda p: p[:, :, 0],
        "vx": lambda p: p[:, :, 1],
        "vy": lambda p: p[:, :, 2],
        "pre": lambda p: p[:, :, 3],
        "torque": None,
        "div-v": None
    }



    parser = argparse.ArgumentParser()
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
        default=-8,
        type=float,
        help="minimum value for colormap",
    )
    parser.add_argument(
        "--vmax",
        default=-4,
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
    file_list = ["chkpt.0006.pk","chkpt.0008.pk","chkpt.0009.pk","chkpt.0188.pk","chkpt.0206.pk","chkpt.0281.pk"]
    sigma_list = []
    for filename in file_list:
        chkpt       = load_checkpoint(filename, require_solver="cbdgam_2d")
        prim        = chkpt["solution"]
        f = fields[args.field](prim).T
        f = np.log10(f)
        sigma_list.append(f)
    mesh = chkpt["mesh"]
    extent = mesh.x0, mesh.x1, mesh.y0, mesh.y1

    label_fontsize = 24
    tick_fontsize = 18
    title_fontsize = 28
    cbar_fontsize = 22
    line_width = 2

    fig, axes = plt.subplots(2, 3, figsize=(16, 12), gridspec_kw={'wspace': 0, 'hspace': 0})

    axes = axes.flatten()

    for i, ax in enumerate(axes):
 # Replace with your actual data
        cm = ax.imshow(
            sigma_list[i],
            origin="lower",
            vmin=args.vmin,            # Replace with args.vmin
            vmax=args.vmax,            # Replace with args.vmax
            cmap=args.cmap,    # Replace with args.cmap
            extent=extent,
            aspect='equal'
        )
            # Hide all ticks initially
        ax.tick_params(
            left=False, right=False, labelleft=False,
            bottom=False, top=False, labelbottom=False
        )

        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)

        # Show y-ticks only on the leftmost column
        if i % 3 == 0:
            ax.tick_params(left=True, labelleft=True, labelsize=tick_fontsize)

        # Show x-ticks only on the bottom row
        if i >= 3:
            ax.tick_params(bottom=True, labelbottom=True, labelsize=tick_fontsize)
        
        if i in [3, 4, 5]:
            ax.set_xlabel(r'$a_0$', fontsize=label_fontsize)
        if i in [0, 3]:
            ax.set_ylabel(r'$a_0$', fontsize=label_fontsize)
            
    cbar = fig.colorbar(cm, ax=axes, orientation='vertical')
    cbar.set_label(r'$log(\Sigma)$', fontsize=cbar_fontsize)
    cbar.ax.tick_params(labelsize=tick_fontsize)

        # Optional: Set tight figure margins
        #plt.subplots_adjust(wspace=0.05, hspace=0.05)
    #plt.tight_layout()
    plt.savefig("D:\Sailfish_new\ProjectSesh\Poster1\CombPlot2.png",transparent=True)
    plt.show()

if __name__ == "__main__":
    exit(main_cbdgam_2d())

