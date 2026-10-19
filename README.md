# Gaussian volume compression for unstructured meshes

Fits a set of 3D Gaussians to the scalar field of a volume mesh (`.vtu`; structured `.vtk` also
loads). Interior models reproduce the field inside the mesh; surface models (`--encode_surface`)
also encode its boundary, rendering samples outside the mesh as empty.

## Setup

```shell
git clone --recursive <repo>
conda env create --file environment.yml
pip install submodules/simple-knn
cd diff-gaussian-rasterization && python setup.py build_ext --inplace
```

Put `diff-gaussian-rasterization` on `PYTHONPATH` to use the in-place build.

## Training and evaluation

```shell
python train.py -s mesh.vtu --cap_max 5015 -m output/model              # interior
python train.py -s mesh.vtu --cap_max 5015 -m output/model --encode_surface
python eval_final.py -s mesh.vtu --models output/model [--surface output/model]
```

`--cap_max` is the final Gaussian count: storage / (ratio x 48 bytes). The defaults are
the final configurations; `run_final.sh` trains and evaluates every final model.

Meshes too large to triangulate on the GPU train on precomputed samples
(`gen_samples.py`, then `--precomputed_samples <prefix>`) and evaluate on a precomputed
tet-interior set (`precompute_tet.py`, then `--eval_set <prefix>`).

Ablations: `--densify_3dgs`, `--use_mcmc`, `--densify_alpha 0` (densification);
`--sample_frac`, `--sample_mode`, `--samples_per_node` (batch subsampling); `--fp_mode`,
`--fn_mode`, `--fnfp_norm`, `--hinge_rel_pow`, `--hinge_freeze` (surface FP/FN terms).
