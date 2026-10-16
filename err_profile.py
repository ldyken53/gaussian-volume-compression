"""Where a model's squared error actually sits: by GT value band, and by local gradient.

Both decompositions are over the FULL volume at native resolution, not a subsample.
Shares are of total squared error, so they say where fixing something would pay.
"""
import argparse, os, time
from types import SimpleNamespace
import numpy as np, pyvista as pv, torch
from gaussian_renderer import render
from scene import GaussianModel

BANDS = [(0.0, 0.0), (0.0, 0.01), (0.01, 0.05), (0.05, 0.10), (0.10, 0.25),
         (0.25, 0.50), (0.50, 0.75), (0.75, 1.01)]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-s", "--source_path", required=True)
    p.add_argument("-m", "--model_path", nargs="+", required=True)
    p.add_argument("--max_scale", type=float, default=0.02)
    p.add_argument("--block", type=int, default=256)
    a = p.parse_args()

    mesh = pv.read(a.source_path)
    nx, ny, nz = mesh.dimensions
    gt_all = mesh.point_data[mesh.point_data.keys()[0]].reshape(nz, ny, nx)
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    d = ((xmax-xmin)/(nx-1), (ymax-ymin)/(ny-1), (zmax-zmin)/(nz-1))
    pipe = SimpleNamespace(debug=False)
    jitter = torch.zeros(3 * a.block**3, dtype=torch.float, device="cuda")

    for mp in a.model_path:
        it = max(int(f.split("_")[-1]) for f in os.listdir(os.path.join(mp, "point_cloud")))
        g = GaussianModel(); g.max_scale = a.max_scale
        with torch.no_grad():
            g.load_ply(os.path.join(mp, "point_cloud", f"iteration_{it}", "point_cloud.ply"), mesh)
            sse_b = torch.zeros(len(BANDS), dtype=torch.float64, device="cuda")
            cnt_b = torch.zeros(len(BANDS), dtype=torch.float64, device="cuda")
            bias_b = torch.zeros(len(BANDS), dtype=torch.float64, device="cuda")
            # gradient quartiles, thresholds fixed from the first block
            gq = None
            sse_g = torch.zeros(4, dtype=torch.float64, device="cuda")
            cnt_g = torch.zeros(4, dtype=torch.float64, device="cuda")
            t0 = time.time()
            for a0 in range(0, nz, a.block):
                for b0 in range(0, ny, a.block):
                    for c0 in range(0, nx, a.block):
                        lo = (xmin + c0*d[0], ymin + b0*d[1], zmin + a0*d[2])
                        g.mins = list(lo)
                        g.maxes = [lo[i] + (a.block-1)*d[i] for i in range(3)]
                        cells = render(g, pipe, jitter, a.block)["cells"]
                        la, lb, lc = min(a.block, nz-a0), min(a.block, ny-b0), min(a.block, nx-c0)
                        cells = cells[:la, :lb, :lc]
                        gtb = torch.as_tensor(np.ascontiguousarray(
                            gt_all[a0:a0+la, b0:b0+lb, c0:c0+lc], dtype=np.float32)).cuda()
                        err = (cells - gtb)
                        e2 = (err**2).double()
                        for i, (lo_v, hi_v) in enumerate(BANDS):
                            # bands must partition: the first is exactly 0, so the
                            # next one starts strictly above it or the zeros (17% of
                            # chameleon) are counted twice.
                            if hi_v == lo_v:
                                m = gtb == 0
                            elif lo_v == 0.0:
                                m = (gtb > 0) & (gtb < hi_v)
                            else:
                                m = (gtb >= lo_v) & (gtb < hi_v)
                            sse_b[i] += e2[m].sum(); cnt_b[i] += m.sum()
                            bias_b[i] += err[m].double().sum()
                        # central-difference gradient magnitude on the GT block interior
                        if min(la, lb, lc) > 2:
                            gx = gtb[1:-1,1:-1,2:] - gtb[1:-1,1:-1,:-2]
                            gy = gtb[1:-1,2:,1:-1] - gtb[1:-1,:-2,1:-1]
                            gz = gtb[2:,1:-1,1:-1] - gtb[:-2,1:-1,1:-1]
                            gm = (gx*gx + gy*gy + gz*gz).sqrt() * 0.5
                            ei = e2[1:-1,1:-1,1:-1]
                            if gq is None:
                                s = gm.flatten()[::97]
                                gq = torch.quantile(s.float(), torch.tensor([.25,.5,.75]).cuda())
                            b = torch.bucketize(gm, gq)
                            for i in range(4):
                                m = b == i
                                sse_g[i] += ei[m].sum(); cnt_g[i] += m.sum()
            tot = float(sse_b.sum())
            n = float(cnt_b.sum())
            print(f"\n=== {mp}  (overall PSNR {-10*np.log10(tot/n):.3f}, {time.time()-t0:.0f}s)")
            print(f"{'GT band':>14s} {'% of volume':>12s} {'% of sq err':>12s} {'band PSNR':>10s} {'mean bias':>11s}")
            for i, (lo_v, hi_v) in enumerate(BANDS):
                c = float(cnt_b[i]);  s = float(sse_b[i])
                if c == 0: continue
                lbl = "exactly 0" if hi_v == lo_v else f"[{lo_v:.2f},{hi_v:.2f})"
                print(f"{lbl:>14s} {100*c/n:12.2f} {100*s/tot:12.2f} "
                      f"{-10*np.log10(max(s/c,1e-20)):10.2f} {float(bias_b[i])/c:+11.5f}")
            tg = float(sse_g.sum()); ng = float(cnt_g.sum())
            print(f"{'|grad GT| quartile':>20s} {'% of sq err':>12s} {'quartile PSNR':>14s}")
            for i in range(4):
                c = float(cnt_g[i]); s = float(sse_g[i])
                if c == 0:
                    print(f"{'Q'+str(i+1):>20s} {'(empty quartile)':>27s}")
                    continue
                print(f"{'Q'+str(i+1):>20s} {100*s/tg:12.2f} {-10*np.log10(max(s/c,1e-20)):14.2f}")
        del g; torch.cuda.empty_cache()

main()
