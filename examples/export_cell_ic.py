"""
Export cell-centered scalar field from a ParaView .vtu file to a regular 2D grid
text file, suitable as AsFem initial condition input.

Usage:
    1. Run: python export_cell_ic.py [--vtu PATH] [--out PATH] [--field NAME]
    2. The output file can be used directly as GridDataIC data_file.

    Default VTU: spinodal2d-Element-results.vtu (must contain <nodes>,
    <connectivity>, and the scalar field as CellData).

Output format (plain text):
    nx ny
    xmin xmax
    ymin ymax
    c(0,0) c(1,0) ... c(nx-1,0)
    c(0,1) c(1,1) ... c(nx-1,1)
    ...

Requirements:
    - Python 3 with numpy
    - The VTU must contain CellData (not PointData) for the field.
    - The mesh must have quadrilateral cells (connectivity length = 4 × n_cells).
    - The number of cells must be a perfect square (nx == ny); otherwise,
      set nx/ny manually via --nx / --ny.

Example:
    python export_cell_ic.py
    python export_cell_ic.py --vtu spinodal2d-Element-results.vtu
    python export_cell_ic.py --vtu my.vtu --out my_ic.txt --field c

Note:
    This script assumes element-based data (one value per cell).
    If your VTU uses PointData (no <nodes> array, equal NumberOfPoints
    and NumberOfCells), use --point-data to extract field values directly
    from point coordinates without computing centroids.
"""

import argparse
import re
import sys
from typing import List

import numpy as np

# ====== Default paths — override via command line ======
VTU_PATH = "spinodal2d-Element-results.vtu"
OUT_PATH = "ic_mesh.txt"
FIELD_NAME = "c"
# =======================================================

# Domain bounds — match your AsFem mesh configuration
XMIN, XMAX = 0.0, 1.0
YMIN, YMAX = 0.0, 1.0


def _get_array(content: str, name: str, vtu_path: str) -> List[str]:
    """Extract a named DataArray from VTU XML content."""
    m = re.search(
        r'<DataArray[^>]*Name="' + name + r'"[^>]*>(.*?)</DataArray>',
        content, re.S,
    )
    if m is None:
        available = re.findall(r'<DataArray[^>]*Name="([^"]+)"', content)
        raise ValueError(
            f"DataArray '{name}' not found in {vtu_path}.\n"
            f"Available DataArrays: {available}\n"
            f"Hint: if this is a PointData VTU (no mesh geometry), "
            f"you may need a different file with CellData."
        )
    return m.group(1).split()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export cell-centered scalar field from VTU to AsFem grid format."
    )
    p.add_argument(
        "--vtu", "-v",
        default=VTU_PATH,
        help=f"Path to ParaView .vtu file (default: {VTU_PATH}).",
    )
    p.add_argument(
        "--out", "-o",
        default=OUT_PATH,
        help=f"Output text file path (default: {OUT_PATH}).",
    )
    p.add_argument(
        "--field", "-f",
        default=FIELD_NAME,
        help=f"Scalar field name in the VTU (default: {FIELD_NAME}).",
    )
    p.add_argument(
        "--nx", type=int, default=None,
        help="Number of cells in x (default: auto-detect from sqrt).",
    )
    p.add_argument(
        "--ny", type=int, default=None,
        help="Number of cells in y (default: auto-detect from sqrt).",
    )
    p.add_argument(
        "--xmin", type=float, default=XMIN,
        help=f"Domain x-min (default: {XMIN}).",
    )
    p.add_argument(
        "--xmax", type=float, default=XMAX,
        help=f"Domain x-max (default: {XMAX}).",
    )
    p.add_argument(
        "--ymin", type=float, default=YMIN,
        help=f"Domain y-min (default: {YMIN}).",
    )
    p.add_argument(
        "--ymax", type=float, default=YMAX,
        help=f"Domain y-max (default: {YMAX}).",
    )
    p.add_argument(
        "--point-data", action="store_true",
        help="VTU uses PointData (no mesh geometry); extract field directly "
             "from point coordinates (assumes structured grid ordering).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.vtu:
        sys.exit("ERROR: --vtu path is required.")
    try:
        with open(args.vtu, "r") as f:
            content = f.read()
    except FileNotFoundError:
        sys.exit(f"ERROR: VTU file not found: {args.vtu}")

    if args.point_data:
        _export_point_data(content, args)
    else:
        _export_cell_data(content, args)


def _export_cell_data(content: str, args: argparse.Namespace) -> None:
    """Export from CellData VTU with mesh geometry (nodes + connectivity)."""
    pts = np.array(_get_array(content, "nodes", args.vtu), dtype=float).reshape(-1, 3)[:, :2]
    conn = np.array(_get_array(content, "connectivity", args.vtu), dtype=int).reshape(-1, 4)
    vals = np.array(_get_array(content, args.field, args.vtu), dtype=float)

    if len(vals) != conn.shape[0]:
        raise ValueError(
            f"'{args.field}' has {len(vals)} values but mesh has {conn.shape[0]} cells.\n"
            "This script expects one value per cell (CellData). "
            "If your field is PointData, try --point-data."
        )

    # Cell centroids
    centroids = pts[conn].mean(axis=1)
    xs = centroids[:, 0]
    ys = centroids[:, 1]

    n_total = len(vals)
    nx = args.nx
    ny = args.ny
    if nx is None or ny is None:
        n_side = int(round(n_total ** 0.5))
        if n_side * n_side != n_total:
            raise ValueError(
                f"{n_total} cells is not a perfect square (sqrt ≈ {n_total**0.5:.1f}). "
                "Set nx, ny manually via --nx / --ny."
            )
        nx = ny = n_side

    _write_output(vals, xs, ys, nx, ny, args)


def _export_point_data(content: str, args: argparse.Namespace) -> None:
    """Export from PointData VTU (no mesh geometry, points = cell centers)."""
    # Try to get coordinates from <Points> or <DataArray Name="nodes">
    try:
        pts = np.array(_get_array(content, "nodes", args.vtu), dtype=float).reshape(-1, 3)[:, :2]
    except ValueError:
        # PointData VTU often has <Points> block with coordinates
        m = re.search(r'<Points>.*?<DataArray[^>]*>(.*?)</DataArray>.*?</Points>', content, re.S)
        if m is None:
            raise ValueError(
                f"No mesh coordinates found in {args.vtu}. "
                "This VTU may not contain spatial position data."
            )
        pts = np.array(m.group(1).split(), dtype=float).reshape(-1, 3)[:, :2]

    vals = np.array(_get_array(content, args.field, args.vtu), dtype=float)

    if len(vals) != pts.shape[0]:
        raise ValueError(
            f"'{args.field}' has {len(vals)} values but there are {pts.shape[0]} points."
        )

    xs = pts[:, 0]
    ys = pts[:, 1]

    n_total = len(vals)
    nx = args.nx
    ny = args.ny
    if nx is None or ny is None:
        # For point data we need to figure out the grid from coordinates
        unique_x = len(set(round(x, 10) for x in xs))
        unique_y = len(set(round(y, 10) for y in ys))
        if unique_x * unique_y != n_total:
            raise ValueError(
                f"Cannot auto-detect grid: {unique_x} unique xs × {unique_y} unique ys "
                f"≠ {n_total} points. Set nx, ny manually via --nx / --ny."
            )
        nx, ny = unique_x, unique_y

    _write_output(vals, xs, ys, nx, ny, args)


def _write_output(
    vals: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    nx: int,
    ny: int,
    args: argparse.Namespace,
) -> None:
    """Sort values into (y, x) grid order and write the AsFem-format text file."""
    if nx * ny != len(vals):
        raise ValueError(
            f"Grid nx*ny={nx*ny} does not match number of values ({len(vals)})."
        )

    order = np.lexsort((xs, ys))
    vals_sorted = vals[order]
    grid = vals_sorted.reshape(ny, nx)

    with open(args.out, "w") as f:
        f.write(f"{nx} {ny}\n")
        f.write(f"{args.xmin:.10f} {args.xmax:.10f}\n")
        f.write(f"{args.ymin:.10f} {args.ymax:.10f}\n")
        for j in range(ny):
            f.write(" ".join(f"{grid[j, i]:.10f}" for i in range(nx)) + "\n")

    print(f"Wrote {args.out}: nx={nx}, ny={ny}, min={grid.min():.6f}, max={grid.max():.6f}")


if __name__ == "__main__":
    main()
