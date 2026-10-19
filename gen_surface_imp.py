"""Surface-encoding training set = interior IMPORTANCE samples + exterior sentinel points.
Per batch b: the exterior (-1) points of impactcombinedbig batch (b mod 100) -- same
draws train.py's own sampler would make -- plus interior points from the importance
set to fill the batch. Layout inside a batch is irrelevant: train.py Morton-sorts at
load and identifies exterior samples by gt == -1."""
import sys, numpy as np, time
U = "/lus/eagle/projects/UIC-HPC/ldyken53/"
src_int, src_ext, out, B = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
Si = np.load(U+src_int+"_samples.npy", mmap_mode="r"); Gi = np.load(U+src_int+"_gt.npy", mmap_mode="r")
Se = np.load(U+src_ext+"_samples.npy", mmap_mode="r"); Ge = np.load(U+src_ext+"_gt.npy", mmap_mode="r")
N = Si.shape[1]; assert Se.shape[1] == N
S = np.lib.format.open_memmap(U+out+"_samples.npy", mode="w+", dtype=np.float32, shape=(B, N, 3))
G = np.lib.format.open_memmap(U+out+"_gt.npy", mode="w+", dtype=np.float32, shape=(B, N))
t = time.time(); tot_ext = 0
for b in range(B):
    ge = np.asarray(Ge[b % Se.shape[0]]); m = ge == -1; k = int(m.sum()); tot_ext += k
    S[b, :k] = np.asarray(Se[b % Se.shape[0]])[m]; G[b, :k] = -1.0
    S[b, k:] = np.asarray(Si[b])[:N-k];           G[b, k:] = np.asarray(Gi[b])[:N-k]
    if b % 25 == 0: S.flush(); G.flush(); print(f"batch {b}: {k} exterior, {time.time()-t:.0f}s", flush=True)
S.flush(); G.flush()
print(f"DONE {out}: {B}x{N}, exterior {tot_ext/(B*N)*100:.2f}%, interior gt mean {np.asarray(G[:5])[np.asarray(G[:5])!=-1].mean():.4f}", flush=True)
