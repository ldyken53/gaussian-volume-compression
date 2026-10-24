# VEGGIES

Volume Encoding Gaussians with Geometry-Integration and Explicit Sampling: compresses a volume
into a set of 3D Gaussians that can be sampled directly at any point, with no decompression step.
Each Gaussian stores a mean, scale, rotation, value and weight (12 floats, 48 bytes), and the
model's value at a point is the weighted average of the Gaussians around it.

There are three types of model (`--kind`):

| type | input | paper | encodes |
|---|---|---|---|
| `structured` | a regular grid: `.vti`, or a legacy `.vtk` with structured points | VEGGIES (structured) | the scalar field |
| `surface` | a mesh: `.vtu`, or any VTK dataset with 3D cells | VEGGIES | the field and the domain; points outside the mesh sample as undefined |
| `interior` | a mesh | VEGGIES+S | the field only; telling whether a point is inside the domain needs the mesh's exterior surface |

Grids train structured models and meshes train surface models by default. Use
`--kind interior` for VEGGIES+S.

## Setup

```shell
git clone --recursive git@github.com:ldyken53/gaussian-volume-compression.git
cd gaussian-volume-compression
conda env create --file environment.yml
conda activate gaussian_splatting
pip install submodules/simple-knn
(cd gvraster && python setup.py build_ext --inplace)
export PYTHONPATH=$PWD/gvraster
```

`gvraster/` holds the CUDA kernels: BVH sampling at arbitrary points (meshes), tile sampling on
grids, and the fused Adam and loss kernels. Building it needs a CUDA toolkit that matches
PyTorch's.

## Your own dataset

1. Put it in a VTK file: a grid as `.vti` (or legacy `.vtk`), a mesh as `.vtu`. A headerless raw
   volume converts with `python tools/raw_to_grid.py data.raw data.vti --shape X Y Z --dtype uint8`.
   If the file holds several arrays, choose one with `--array NAME`.
2. Train:
   ```shell
   python train.py -s data.vti -m output/model --ratio 256
   ```
   `--ratio R` sizes the model at (volume bytes) / (48 R) Gaussians, counting 4 bytes per grid
   node, or a mesh's points, cells and scalar. `--cap_max N` asks for exactly N Gaussians.
3. Evaluate:
   ```shell
   python eval.py -s data.vti -m output/model --continuous 2    # grids
   python eval.py -s mesh.vtu -m output/model                   # meshes
   ```
4. Sample it:
   ```python
   from veggies import load_model, train

   train("data.vti", "output/model", ratio=256)   # the same as the command line
   model = load_model("output/model")
   values = model.sample(points)                  # (N, 3) points; NaN where undefined
   ```
   Training scales the volume into the unit cube with values in [0, 1] and records the transform
   in `veggies.json`, so `sample` takes and returns the volume's own coordinates and values.
   Grids that are already scaled can pass `--is_scaled`.

Meshes too large to triangulate on the GPU can train on precomputed samples
(`tools/gen_samples.py`, then `--precomputed_samples PREFIX`) and evaluate on precomputed points
(`tools/precompute_tet.py`, then `eval.py --eval_set PREFIX`).

A model directory holds `point_cloud/iteration_<N>/point_cloud.ply` (the Gaussians, saved as
their parameter average), `max_scale` and `render_rule` (needed to sample it), `veggies.json`
(its type and normalization), `cfg_args` (the training options) and `train_seconds.txt`.

## Sampling

With V = Σ w G v and I = Σ w G (the accumulated influence) over the Gaussians at a point, the
point is undefined where I ≤ T and otherwise has the value V / max(I, τ1). (T, τ1) is (0.002, 0.01)
for structured and interior models and (0.01, 0.01) for surface models; `--defined_threshold`
and `--floor` override them. A Gaussian reaches a point only where w G ≥ 10⁻⁴. Models train with
T = 0 and store their rule in `render_rule`.

## Training

Every model trains for 5,000 iterations. Each Adam step updates a parameter average (decay 0.995,
from iteration 1,000), and that average is what gets saved. Sample-error-based densification adds
Gaussians at samples drawn with probability ∝ error^2.5, growing the model about 1.5× per event
until the target count. Dead Gaussians (weight below τ2 = 0.005) are reused for new ones.
`--densify 3dgs` and `--densify mcmc` select the alternatives the paper compares against.

What differs by type (set from the type unless given):

| | structured | interior | surface |
|---|---|---|---|
| seeds (`--fraction`, `--init_share_max`, `--init_max`) | 10⁻⁵ of the nodes, at most 25% of the target and 20,000 | every node, at most 50% of the target and 20,000 | every node, at most 50% of the target and 20,000 |
| samples per step | one of 400 precomputed jittered 128³ lattices (`--jitter_gt`) | ~0.25 per mesh node (`--samples_per_node`) | the same number, replacing 20% with points just outside the mesh and 10% just inside its surface (`--interior_skin`) |
| reconstruction loss | L2 over every lattice cell | L2 over samples in the mesh | L2 over samples in the mesh |
| FN / FP loss | log barrier on I (λ_FN 10⁻⁴) | log barrier on I | hinges around T, FN inside and FP outside, scaled by the L1 data loss (λ_FN 1200, λ_FP 12) |

A Gaussian's scale is softly capped at max(0.02, 1.25 (V/N)^⅓), where V is the bounding-box
volume and N the target count. `python train.py -h` lists every option.

## Code

| path | |
|---|---|
| `train.py`, `eval.py` | command-line entry points |
| `veggies/model.py` | the model: parameters, activations, Adam with the parameter average, row edits, ply files |
| `veggies/sampling.py` | sampling a model at points (BVH) or on a lattice (tiles) |
| `veggies/samples.py` | what each type trains on |
| `veggies/losses.py` | reconstruction, FN and FP losses |
| `veggies/densify.py` | densification schedule and strategies |
| `veggies/volume.py` | reading and normalizing volumes, seeds, model size |
| `veggies/config.py` | options and per-type defaults |
| `veggies/train.py`, `veggies/evaluate.py`, `veggies/api.py` | training, evaluation, `load_model` |
| `gvraster/` | CUDA kernels |
| `tools/` | raw-volume conversion, precomputed sample sets |
| `run_final.sh` | trains and evaluates the paper's final models |

Older option names still work: `--encode_surface` (`--encode_domain`), `--fn_reg` / `--fp_reg`
(`--lambda_fn` / `--lambda_fp`), `--min_weight` (`--prune_threshold`), `--occ_t`
(`--defined_threshold`), `--denom_floor` (`--floor`), `--use_mcmc` / `--densify_3dgs`
(`--densify mcmc` / `3dgs`). A mesh given without `--kind` used to train an interior model; it now
trains a surface model.
