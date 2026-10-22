# Gaussian volume compression

Fits a set of 3D Gaussians to the scalar field of a volume. The Gaussians are the compressed
volume: a model stores 12 floats per Gaussian (position, scale, rotation, value, weight) and
reconstructs the field at any point by blending the Gaussians around it.

One pipeline trains three types of model:

| type | input | trains on | renderer |
|---|---|---|---|
| `structured` | a regular grid (legacy `.vtk` STRUCTURED_POINTS) | jittered 128³ lattices over the grid, against the trilinear interpolant | tile rasterizer (`gvraster.tile`) |
| `interior` | a volume mesh (`.vtu`) | points sampled inside the mesh | BVH over the sample points (`gvraster.bvh`) |
| `surface` | a volume mesh (`.vtu`) | interior points, plus a band just outside the boundary that must render empty | BVH over the sample points (`gvraster.bvh`) |

The type follows the input: a `.vtk` grid trains a structured model, a `.vtu` mesh an interior
model, and `--encode_surface` a surface model. `--kind` overrides the choice, for example
`--kind interior` to fit a `.vtk` grid as a mesh. All three types share the Gaussian model, the
training loop, densification, the parameter average and the render rule; the defaults that
differ are listed under [Model types](#model-types).

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

`gvraster/` is the CUDA side, built once into three extension modules: the two renderers
(`gvraster._bvh`, `gvraster._tile`) and the Adam and loss kernels every model trains with
(`gvraster._ops`). Building it needs a CUDA toolkit that matches PyTorch's CUDA version and a
compatible C++ compiler. Run every script from the repository root with `gvraster/` on
`PYTHONPATH`.

## Training

```shell
# a regular grid: --ratio sets the Gaussian count to nodes / (12 x ratio)
python train.py -s volume.vtk --is_scaled --ratio 1024 -m output/grid
# a volume mesh: --cap_max is the final Gaussian count, storage / (ratio x 48 bytes)
python train.py -s mesh.vtu --cap_max 5015 -m output/interior
python train.py -s mesh.vtu --cap_max 5015 -m output/surface --encode_surface
```

`--is_scaled` says a grid is already in the unit cube with values in [0, 1]; otherwise it is
normalised on load (meshes always are). `--seed` (default `$SEED`, else 0) seeds every random
choice. The defaults are the final configurations: `run_final.sh` trains and evaluates every
final model. `train.py -h` lists every option, grouped by the model types it applies to;
options for grids only or meshes only are refused for the other.

The first run on a mesh probes the ground truth of its sample batches, which takes minutes on
large meshes, and caches it in `sample_cache/` (`--sample_cache_dir`); later runs on the same
mesh reuse it.

A model directory holds:

| file | |
|---|---|
| `point_cloud/iteration_<N>/point_cloud.ply` | the Gaussians: the parameter average at iteration N (the last one, plus `--save_iterations`) |
| `max_scale` | the scale cap the model was trained with; rendering needs it |
| `render_rule` | the model's (T, F), see [Render rule](#render-rule) |
| `cfg_args` | the training arguments |
| `train_seconds.txt` | training time, saves excluded |

## Evaluation

```shell
python eval.py -s volume.vtk --is_scaled --continuous 2 -m output/grid
python eval.py -s mesh.vtu -m output/interior output/surface --surface output/surface
```

`eval.py` scores any number of models against their source volume, each under its own
`max_scale` and `render_rule`. `--out` also writes the results as JSON.

- **Grids** (a `.vtk` source): every native grid node is rendered. The output reports `psnr`
  (an empty node renders as -1), `psnr_covered` (non-empty nodes only), `covered` (the share
  of non-empty nodes) and foreground/background splits. `--continuous T` adds
  `psnr_continuous`, the metric of record for structured models: PSNR against the trilinear
  interpolant at T jittered 128³ lattices. Prints one `RESULT` line per model.
- **Meshes** (a `.vtu` source): models are scored on 1M fresh points inside the mesh (or a
  precomputed `--eval_set`). The output reports `psnr_gt` (an empty render counts as 0),
  `psnr_both` (empty renders excluded) and `fn` (points inside the mesh that render empty).
  Models also listed under `--surface` report `fp_pct`, the share of points just outside the
  surface that render occupied. Prints one JSON line per model.

`--protocol grid|mesh` overrides the choice made from the file extension.

## Render rule

Every model renders the same way. At a point, let S = Σ w G v and aw = Σ w G, both summed over
the Gaussians. The point is empty (-1) where aw <= T; otherwise its value is S / max(aw, F).
Each model stores its (T, F) in `<model>/render_rule`: (0.002, 0.01) for structured and
interior models, (0.01, 0.01) for surface models. `--occ_t` and `--denom_floor` override them.
Every model trains with the soft render (T = 0). Models without a `render_rule` file predate
the file: they render with (0.01, 0.01) if they are surface models, else (0, 0.01).

## Model types

`train.py` sets these per type. A flag given on the command line overrides them.

| | structured | interior | surface |
|---|---|---|---|
| seeds (`--fraction`, `--init_share_max`) | 1e-5 of the grid nodes, at most 25% of the final count | every mesh node, at most 50% | every mesh node, at most 50% |
| value loss | L2 on every lattice cell | L2 on the samples inside the mesh | L2 |
| coverage loss | log barrier on the accumulated weight (`--fn_mode log`, `--fn_reg 1e-4`, `--fn_thresh`) | log barrier | the log barrier, plus FN and FP hinges around T scaled by the L1 data loss (`--fn_mode abs`, `--fn_reg 1200`, `--fp_reg 12`, `--fp_mode hinge`) |
| training samples | `--jitter_gt` precomputed lattices (default 400) | `--samples_per_node` per step, or `--sample_frac` of each batch | the same, plus 20% just outside the surface and `--interior_skin` (10%) just inside it |
| MCMC `--weight_reg` | 0.01 | 1e-5 | 1e-5 |
| 3DGS `--densify_grad_threshold` | 0.0002 | 0 | 0 |

Every type also caps its seeds at `--init_max` (20000).

Densification places each new Gaussian at a sample drawn with probability proportional to
error^`--densify_alpha`, reusing dead Gaussians' rows; `--densify_growth` sets how many events
it takes. Two alternatives are ablations: `--use_mcmc` (relocate/add) and `--densify_3dgs`
(clone/split by position gradient). The saved model is the parameter average (`--ema`, from
`--ema_from`); `--ema 0` ships the last iterate. A Gaussian's scale is capped at
max(`--max_scale`, `--max_scale_k` (V / N)^(1/3)), where V is the bounding-box volume and N
the final count.

Meshes too large to triangulate on the GPU train on precomputed samples (`gen_samples.py`, then
`--precomputed_samples <prefix>`; add `--surface` for a surface set). They are evaluated on a
precomputed set of points inside the tetrahedra (`precompute_tet.py`, then
`eval.py --eval_set <prefix>`). `raw_to_vtk.py` converts a raw volume into a structured `.vtk`.

## Layout

| path | |
|---|---|
| `train.py`, `eval.py` | training and evaluation, every model type |
| `scene/gaussian_model.py` | the model: parameters and activations, the ply layout, the fused Adam step with the parameter average, densification and pruning |
| `scene/samples.py` | what a model trains on: mesh sample batches (`MeshSamples`) and grid lattices (`GridSamples`) |
| `scene/dataset_readers.py` | `.vtk` and `.vtu` readers and the seed points |
| `scene/densify.py`, `scene/model_files.py` | the densification schedule; the per-model files and the scale cap |
| `gaussian_renderer/` | `render_points` (BVH) and `render_grid` (tile) under the render rule, and the surface margin term |
| `gvraster/` | the CUDA extensions: `csrc/bvh`, `csrc/tile`, `csrc/ops` |
| `arguments/`, `utils/` | the shared option groups; rotation helpers and seeding |
| `run_final.sh` | trains and evaluates every final model |
| `submodules/simple-knn` | nearest-neighbour distances for the seed scales |

The code borrows from [3D Gaussian Splatting for Real-Time Radiance Field
Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/); see `LICENSE` and
`NOTICE`.
