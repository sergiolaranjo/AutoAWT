"""AutoAWT Visualization Tools

Interactive 3D visualization of wall thickness results.

Supports multiple backends:
    1. ParaView (external) — open .vtp files directly
    2. PyVista (Python) — interactive 3D in Python
    3. Matplotlib (fallback) — static 3D scatter + histograms

Usage:
    # Convert existing results to ParaView format
    python visualize.py --convert /path/to/Results

    # Interactive visualization with PyVista
    python visualize.py --show /path/to/Results

    # Generate static report (matplotlib)
    python visualize.py --report /path/to/Results

    # Open in ParaView directly
    python visualize.py --paraview /path/to/Results
"""

import os
import sys
import argparse
import numpy as np


def load_plt_data(plt_path):
    """Load vertex + thickness data from a PLT file.

    Returns:
        points: Nx3 array of (x, y, z) coordinates
        thickness: N array of thickness values (or None)
        faces: Mx3 array of triangle indices (or None for point clouds)
    """
    points = []
    thickness = []
    faces = []
    n_verts = 0
    n_elems = 0
    reading_faces = False
    vert_count = 0

    with open(plt_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('VARIABLES'):
                continue

            if line.startswith('ZONE'):
                # Parse zone header
                for token in line.replace(',', ' ').split():
                    tok = token.upper()
                    if tok.startswith('I='):
                        n_verts = int(tok[2:])
                    elif tok.startswith('N='):
                        n_verts = int(tok[2:])
                    elif tok.startswith('E='):
                        n_elems = int(tok[2:])
                continue

            parts = line.split()
            if reading_faces and len(parts) >= 3:
                faces.append([int(parts[0])-1, int(parts[1])-1, int(parts[2])-1])
                if len(faces) >= n_elems:
                    break
                continue

            if len(parts) >= 3:
                try:
                    points.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    if len(parts) >= 4:
                        thickness.append(float(parts[3]))
                    vert_count += 1
                except ValueError:
                    continue

            if n_verts > 0 and vert_count >= n_verts and n_elems > 0:
                reading_faces = True

    points = np.array(points, dtype=np.float32) if points else np.zeros((0, 3))
    thickness = np.array(thickness, dtype=np.float32) if thickness else None
    faces = np.array(faces, dtype=np.int32) if faces else None

    return points, thickness, faces


def find_result_files(results_dir):
    """Find all relevant result files in a Results directory."""
    files = {}
    for f in os.listdir(results_dir):
        fp = os.path.join(results_dir, f)
        if f == 'WT-endo.plt':
            files['endo_plt'] = fp
        elif f.startswith('WT(projected)') and f.endswith('.plt'):
            files['projected_plt'] = fp
        elif f.startswith('WT(projected)') and f.endswith('.stl'):
            files['projected_stl'] = fp
        elif f == 'surface_mesh.stl':
            files['surface_stl'] = fp
        elif f == 'WT-endo.vtp':
            files['endo_vtp'] = fp
        elif f.startswith('WT(projected)') and f.endswith('.vtp'):
            files['projected_vtp'] = fp
        elif f == 'surface_mesh.vtp':
            files['surface_vtp'] = fp
    return files


# ============================================================
# Convert results to ParaView format
# ============================================================

def convert_to_paraview(results_dir):
    """Convert all PLT/STL files in results directory to VTP."""
    from io_formats import plt_to_vtp, stl_to_vtp

    files = find_result_files(results_dir)
    converted = []

    if 'endo_plt' in files:
        vtp = plt_to_vtp(files['endo_plt'])
        converted.append(vtp)

    if 'projected_plt' in files:
        vtp = plt_to_vtp(files['projected_plt'])
        converted.append(vtp)

    if 'surface_stl' in files:
        thickness_plt = files.get('endo_plt')
        vtp = stl_to_vtp(files['surface_stl'], thickness_plt=thickness_plt)
        converted.append(vtp)

    if converted:
        print(f"\nConverted {len(converted)} files to VTP format.")
        print("Open in ParaView:")
        for f in converted:
            print(f"  paraview {f}")
    else:
        print("No result files found to convert.", file=sys.stderr)

    return converted


# ============================================================
# PyVista Interactive Visualization
# ============================================================

def show_pyvista(results_dir):
    """Interactive 3D visualization using PyVista."""
    try:
        import pyvista as pv
    except ImportError:
        print("PyVista not installed. Install with: pip install pyvista", file=sys.stderr)
        print("Falling back to matplotlib...", file=sys.stderr)
        return show_matplotlib(results_dir)

    files = find_result_files(results_dir)

    # Check for VTP files first, convert if needed
    if 'endo_vtp' not in files and 'endo_plt' in files:
        print("Converting to VTP for visualization...")
        convert_to_paraview(results_dir)
        files = find_result_files(results_dir)

    plotter = pv.Plotter(shape=(1, 2), title="AutoAWT - Wall Thickness")

    # Left panel: Endo surface with thickness
    plotter.subplot(0, 0)
    plotter.add_text("Endocardium Thickness", font_size=12)

    if 'endo_vtp' in files:
        mesh = pv.read(files['endo_vtp'])
        if 'Thickness(mm)' in mesh.point_data:
            plotter.add_mesh(mesh, scalars='Thickness(mm)',
                             cmap='jet', point_size=3,
                             scalar_bar_args={'title': 'Thickness (mm)'})
        else:
            plotter.add_mesh(mesh, color='red', point_size=3)
    elif 'endo_plt' in files:
        points, thickness, _ = load_plt_data(files['endo_plt'])
        cloud = pv.PolyData(points)
        if thickness is not None:
            cloud['Thickness(mm)'] = thickness
            plotter.add_mesh(cloud, scalars='Thickness(mm)',
                             cmap='jet', point_size=3,
                             scalar_bar_args={'title': 'Thickness (mm)'})
        else:
            plotter.add_mesh(cloud, color='red', point_size=3)

    # Right panel: Projected mesh
    plotter.subplot(0, 1)
    plotter.add_text("Surface Mesh", font_size=12)

    if 'projected_vtp' in files:
        mesh = pv.read(files['projected_vtp'])
        if 'Thickness(mm)' in mesh.point_data:
            plotter.add_mesh(mesh, scalars='Thickness(mm)', cmap='jet',
                             scalar_bar_args={'title': 'Thickness (mm)'})
        else:
            plotter.add_mesh(mesh, color='lightblue')
    elif 'surface_stl' in files:
        mesh = pv.read(files['surface_stl'])
        plotter.add_mesh(mesh, color='lightblue', opacity=0.7)

    plotter.link_views()
    plotter.show()


# ============================================================
# Matplotlib Fallback Visualization
# ============================================================

def show_matplotlib(results_dir):
    """Static visualization using matplotlib (no external 3D libs needed)."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib",
              file=sys.stderr)
        return

    files = find_result_files(results_dir)

    if 'endo_plt' not in files:
        print("No WT-endo.plt found in results directory.", file=sys.stderr)
        return

    points, thickness, _ = load_plt_data(files['endo_plt'])
    if thickness is None or len(thickness) == 0:
        print("No thickness data found.", file=sys.stderr)
        return

    # Filter out zeros for statistics
    nonzero = thickness[thickness > 0.1]

    fig = plt.figure(figsize=(16, 5))

    # 1. 3D scatter plot
    ax1 = fig.add_subplot(131, projection='3d')
    # Subsample for performance
    n = len(points)
    step = max(1, n // 5000)
    idx = slice(None, None, step)

    sc = ax1.scatter(points[idx, 0], points[idx, 1], points[idx, 2],
                     c=thickness[idx], cmap='jet', s=1, vmin=0,
                     vmax=np.percentile(nonzero, 95) if len(nonzero) > 0 else 5)
    plt.colorbar(sc, ax=ax1, label='Thickness (mm)', shrink=0.6)
    ax1.set_title('Endocardium Thickness')
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')

    # 2. Histogram
    ax2 = fig.add_subplot(132)
    if len(nonzero) > 0:
        ax2.hist(nonzero, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
        ax2.axvline(np.mean(nonzero), color='red', linestyle='--',
                    label=f'Mean: {np.mean(nonzero):.2f} mm')
        ax2.axvline(np.median(nonzero), color='orange', linestyle='--',
                    label=f'Median: {np.median(nonzero):.2f} mm')
        ax2.legend()
    ax2.set_xlabel('Thickness (mm)')
    ax2.set_ylabel('Count')
    ax2.set_title('Thickness Distribution')

    # 3. Statistics table
    ax3 = fig.add_subplot(133)
    ax3.axis('off')
    stats = [
        ['Total vertices', f'{len(thickness):,}'],
        ['Non-zero vertices', f'{len(nonzero):,} ({100*len(nonzero)/len(thickness):.0f}%)'],
        ['Mean (non-zero)', f'{np.mean(nonzero):.2f} mm' if len(nonzero) > 0 else 'N/A'],
        ['Median', f'{np.median(nonzero):.2f} mm' if len(nonzero) > 0 else 'N/A'],
        ['Std Dev', f'{np.std(nonzero):.2f} mm' if len(nonzero) > 0 else 'N/A'],
        ['Min (non-zero)', f'{np.min(nonzero):.2f} mm' if len(nonzero) > 0 else 'N/A'],
        ['Max', f'{np.max(nonzero):.2f} mm' if len(nonzero) > 0 else 'N/A'],
        ['P5', f'{np.percentile(nonzero, 5):.2f} mm' if len(nonzero) > 0 else 'N/A'],
        ['P95', f'{np.percentile(nonzero, 95):.2f} mm' if len(nonzero) > 0 else 'N/A'],
    ]
    table = ax3.table(cellText=stats, colLabels=['Metric', 'Value'],
                      loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax3.set_title('Statistics')

    plt.tight_layout()
    out_path = os.path.join(results_dir, 'thickness_report.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Report saved: {out_path}")
    plt.close()


# ============================================================
# Open in ParaView
# ============================================================

def open_paraview(results_dir):
    """Open VTP files directly in ParaView."""
    import subprocess
    import shutil

    files = find_result_files(results_dir)

    # Convert if needed
    if not any(k.endswith('_vtp') for k in files):
        print("No VTP files found. Converting...")
        convert_to_paraview(results_dir)
        files = find_result_files(results_dir)

    vtp_files = [v for k, v in files.items() if k.endswith('_vtp')]
    if not vtp_files:
        print("No VTP files available.", file=sys.stderr)
        return

    # Find ParaView
    paraview = shutil.which('paraview')
    if paraview is None:
        print("ParaView not found in PATH.", file=sys.stderr)
        print("Install ParaView from: https://www.paraview.org/download/")
        print("\nYou can open these files manually in ParaView:")
        for f in vtp_files:
            print(f"  {f}")
        return

    print(f"Opening {len(vtp_files)} files in ParaView...")
    subprocess.Popen([paraview] + vtp_files)


# ============================================================
# Main CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='AutoAWT Visualization Tools',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize.py --convert Results/    # Convert to ParaView VTP
  python visualize.py --show Results/       # Interactive 3D (PyVista)
  python visualize.py --report Results/     # Static PNG report
  python visualize.py --paraview Results/   # Open in ParaView
        """
    )

    parser.add_argument('--convert', type=str, metavar='DIR',
                        help='Convert PLT/STL results to VTP for ParaView')
    parser.add_argument('--show', type=str, metavar='DIR',
                        help='Interactive 3D visualization (PyVista)')
    parser.add_argument('--report', type=str, metavar='DIR',
                        help='Generate static PNG report (matplotlib)')
    parser.add_argument('--paraview', type=str, metavar='DIR',
                        help='Open results in ParaView')

    args = parser.parse_args()

    if args.convert:
        convert_to_paraview(args.convert)
    elif args.show:
        show_pyvista(args.show)
    elif args.report:
        show_matplotlib(args.report)
    elif args.paraview:
        open_paraview(args.paraview)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
