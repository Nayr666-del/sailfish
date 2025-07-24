import argparse
import pickle as pk
import numpy as np

class FixNumpyCoreUnpickler(pk.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)



def load_checkpoint(filename, require_solver=None):
    with open(filename, "rb") as f:
        chkpt = FixNumpyCoreUnpickler(f).load()
    return chkpt


def ZoomIn2D(chkpt, n, size=None):
    sink_min_radius = 0.04
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


    chkpt["solution"] = prim_zoom.repeat(n, axis=0).repeat(n, axis=1)
    chkpt["driver"] = chkpt["driver"]._replace(resolution=new_resolution * n)
    chkpt["mesh"] = mesh_zoom._replace(ni=mesh_zoom.ni * n, nj=mesh_zoom.nj * n)
    chkpt["model_parameters"]["sink_radius"] = chkpt["model_parameters"]["sink_radius"] * size / chkpt["model_parameters"]["domain_radius"]
    chkpt["model_parameters"]["domain_radius"] = size
    chkpt["model_parameters"]["softening_length"] = chkpt["model_parameters"]["sink_radius"]
    # Ryan: ensure that the minimum sink radius is equal to the gravitational radii
    if chkpt["model_parameters"]["sink_radius"] < sink_min_radius:
        chkpt["model_parameters"]["sink_radius"] = sink_min_radius
    # Update point masses to the new mesh
    # Needs to change the sink radii, ask David tomorrow
    # chkpt["point_masses"] = PointMass() -> sink source terms, etc.
    # Potentially need to update the circumbinary_disk.py setups
    # Driver terms?
    return chkpt


def main_zoom_in_2d():
    fields = {
        "sigma": lambda p: p[:, :, 0],
        "vx": lambda p: p[:, :, 1],
        "vy": lambda p: p[:, :, 2],
        "pre": lambda p: p[:, :, 3],
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", type=str, nargs="+")
    parser.add_argument(
        "--size",
        "-s",
        default=10.0,
        type=float,
        help="size of the zoomed-in region in code units",
    )
    parser.add_argument(
        "--upsample_scale",
        "-u",
        default=2,
        type=int,
        help="scale factor for upsampling the mesh and solution",
    )
    parser.add_argument(
        "--Outputs",
        "-o",
        default=None,
        type=str,
        help="Where to save zoomed-in chkpts",
    )

    args = parser.parse_args()
    for filename in args.checkpoints:
        chkpt       = load_checkpoint(filename, require_solver="cbdgam_2d")
        chkpt_new = ZoomIn2D(chkpt, args.upsample_scale, size=args.size)
        print(f"Zoomed-in mesh:")
        print(chkpt_new["mesh"])
        print(chkpt_new["model_parameters"]["domain_radius"])
        if args.Outputs is not None:
            outfile = filename.replace('.pk', '_zoomed.pk')
            with open(outfile, "wb") as f:
                pk.dump(chkpt_new, f)
            print(f"Saved zoomed-in checkpoint to {outfile}")
        else:
            print("Zoomed-in checkpoint not saved, use --Outputs to specify a directory.")

if __name__ == "__main__":
    exit(main_zoom_in_2d())



  
