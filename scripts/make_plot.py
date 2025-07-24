import numpy as np
from pathlib import Path
import argparse
import pickle as pk
import os
import sys
import sailfish
import subprocess


class FixNumpyCoreUnpickler(pk.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)

def load_checkpoint(filename, require_solver=None):
    with open(filename, "rb") as f:
        chkpt = FixNumpyCoreUnpickler(f).load()
    return chkpt


def file_load(indir, plots_outdir, follow, radius, vmin, vmax):
    file_count        = 0
    current_path_name = Path().resolve()

    if current_path_name.name != "sailfish":
        raise RuntimeError(f"Script was designed to be run from 'sailfish', but you're in: {current_path_name}")

    output_dir = Path(plots_outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for name in sorted(Path(indir).iterdir()):
        file_count += 1

        plot_script = str(current_path_name / "plot.py")
        if follow:
            plot_args = [
                "python", plot_script,
                name,
                "-l",           
                "-c", str(radius),
                "--vmin", str(vmin),
                "--vmax", str(vmax),
                "-o", str(plots_outdir)
            ]
        else:
            plot_args = [
                "python", plot_script,
                name,
                "-l",           
                "--radius", str(radius),
                "--vmin", str(vmin),
                "--vmax", str(vmax),
                "-o", str(plots_outdir)
            ]

        subprocess.run(plot_args, check=True)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', default='', help='Checkpoint file directory.', required=True)
    parser.add_argument('--outdir', default='movie', help='Output movie directory.')
    parser.add_argument('--follow', default=False, help="Follow the BH")
    parser.add_argument('--radius', default=5, help='Radius of the plot.')
    parser.add_argument('--vmin', default=-10, help='Minimum value for the color scale.')
    parser.add_argument('--vmax', default=-5, help='Maximum value for the color scale.')

    args = parser.parse_args()

    file_load(args.indir, args.outdir, args.follow, args.radius, args.vmin, args.vmax)
