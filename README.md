# Gaussian volume compression

Two pipelines fit a set of 3D Gaussians to the scalar field of a volume:

| pipeline | directory | input | rasterizer (Python package) | evaluation |
|---|---|---|---|---|
| unstructured | repository root | volume meshes (`.vtu`) | BVH over sample points (`bvh_diff_gaussian_rasterization`) | `eval_final.py` |
| structured | `structured/` | regular grids (legacy `.vtk` STRUCTURED_POINTS) | tile rasterizer over the grid (`diff_gaussian_rasterization`) | `structured/full_psnr.py` |

Code both pipelines share lives in `common/gvcommon/`. Both pipelines have a `train.py`,
`scene/`, `gaussian_renderer/`, `utils/` and `arguments/`: run each from its own directory with
its own extension and `common/` on `PYTHONPATH`. Never put both pipeline roots on `sys.path`,
or one pipeline's modules stand in for the other's.

```shell
git clone --recursive <repo>
conda env create --file environment.yml
pip install submodules/simple-knn          # both pipelines import the installed simple_knn
(cd diff-gaussian-rasterization && python setup.py build_ext --inplace)
(cd structured/diff-gaussian-rasterization && python setup.py build_ext --inplace)

# unstructured, from the repository root
export PYTHONPATH=$PWD/diff-gaussian-rasterization:$PWD/common
python train.py -s mesh.vtu --cap_max 5015 -m output/model [--encode_surface]
python eval_final.py -s mesh.vtu --models output/model [--surface output/model]

# structured, from structured/
cd structured && export PYTHONPATH=$PWD/diff-gaussian-rasterization:$PWD/../common
python train.py -s volume.vtk --is_scaled --ratio 1024 -m output/model
python full_psnr.py -s volume.vtk --is_scaled --continuous 2 -m output/model
```

`structured/README.md` describes the structured pipeline. Its `submodules/simple-knn` is not
kept: it pinned the same commit as `submodules/simple-knn`, which both pipelines use.

## Unstructured pipeline

Fits a set of 3D Gaussians to the scalar field of a volume mesh (`.vtu`; structured `.vtk` also
loads). Interior models reproduce the field inside the mesh; surface models (`--encode_surface`)
also encode its boundary, rendering samples outside the mesh as empty.

### Setup

```shell
git clone --recursive <repo>
conda env create --file environment.yml
pip install submodules/simple-knn
cd diff-gaussian-rasterization && python setup.py build_ext --inplace
```

Put `diff-gaussian-rasterization` (the in-place build) and `common` on `PYTHONPATH`.

### Training and evaluation

```shell
python train.py -s mesh.vtu --cap_max 5015 -m output/model              # interior
python train.py -s mesh.vtu --cap_max 5015 -m output/model --encode_surface
python eval_final.py -s mesh.vtu --models output/model [--surface output/model]
```

`--cap_max` is the final Gaussian count: storage / (ratio x 48 bytes). The defaults are
the final configurations; `run_final.sh` trains and evaluates every final model.

### Render rule

Every model renders the same way: at a point, with S = sum w G v and aw = sum w G over the
Gaussians, the point is empty where aw <= T and otherwise has the value S / max(aw, F).
Each model stores its (T, F) in `<model>/render_rule`: interior models (0.002, 0.01), surface
models (0.01, 0.01); `--occ_t` and `--denom_floor` override them. All models train with the
soft render (T = 0). Models without a `render_rule` file predate it and use (0.01, 0.01) if
they are surface models, else (0, 0.01).

Meshes too large to triangulate on the GPU train on precomputed samples
(`gen_samples.py`, then `--precomputed_samples <prefix>`; add `--surface` for an
`--encode_surface` set with train.py's exterior band and inner skin) and evaluate on a
precomputed tet-interior set (`precompute_tet.py`, then `--eval_set <prefix>`).

Ablations: `--densify_3dgs`, `--use_mcmc`, `--densify_alpha 0` (densification);
`--sample_frac`, `--sample_mode`, `--samples_per_node` (batch subsampling); `--fp_mode`,
`--fn_mode`, `--fnfp_norm`, `--hinge_rel_pow`, `--hinge_freeze` (surface FP/FN terms).
