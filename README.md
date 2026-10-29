# ShowerLM full generator (0.6.4c)

Generates in-ice Cherenkov light showers in the format of the CORSIKA 8 full-shower voxel cache: photons per voxel
in a 20 m × 20 m × 20 m box (960³ voxels of 2.083 cm), light emitted within the first 80 ns. This is the model behind
the 0.6.4c evaluation (tz skeleton, group detail model, t-cap stop). The trained models and the small prior tables
are in this folder; no training data is needed.

It is meant for tests, comparisons and method development, not for physics analysis: see *How good it is* below.

## Usage

```bash
pip install -r requirements.txt
python generate.py --n 10 --out showers --seed 0
```

| option | default | meaning |
|---|---|---|
| `--n` | 10 | number of showers |
| `--out` | `showers` | output folder (must not exist) |
| `--seed` | 0 | random seed; the same seed, batch sizes and GPU type give the same showers |
| `--device` | `auto` | sets the batch sizes: `a40` (skeleton 256, detail 32), `1080` (32, 8), `cpu` (4, 2); `auto` picks `a40` on a GPU with more than 20 GB, `1080` on a smaller GPU, `cpu` without one |
| `--skel-batch`, `--det-batch` | by `--device` | override the batch sizes |
| `--render` | `rvC` | voxel renderer: `rvC` is the one the evaluation uses; `v06` is the plain 0.6 renderer |
| `--no-kv` | off | switch off the KV cache of the attribute decoder (slower, same distribution) |

The sampling settings are fixed in `run.json` (the 0.6.4c `tz_cap` settings); `generate.py` passes them to
`showerlm.sample06`.

Measured time per shower (model loading not included):

| hardware | batches (skeleton, detail) | time per shower |
|---|---|---|
| A40, KV cache | 256, 32 | 10.0 s (skeleton 1.8 s + detail 8.2 s), median of 1,024 showers |
| GTX 1080, KV cache | 32, 8 | 24.9 s (skeleton 8.2 s + detail 16.8 s), median of 1,024 showers |
| CPU, 8 threads (login node) | 1, 1 | about 113 s (skeleton 73 s + detail 40 s), one shower |
| CORSIKA 8, for comparison | | 17.7 s per shower (CPU) |

On a busy GPU node the GTX 1080 time rose to about 40 s per shower. Self-test of this package: on a GTX 1080, 8 showers from `generate.py` are token-identical to the 0.6.4c
evaluation code run with the same seed and the same float16 weights. With the original float32 weights the same seed
gives different showers (float16 rounding changes the sampling path after a few hundred to a few thousand tokens);
they are draws of the same model, not copies. The A40 test is still running.

Memory: 2–5 GB on the GPU, under 8 GB of host memory. Use a GPU; the CPU path is for tests.

To view the showers, open `visualize.ipynb`.

## HTCondor

Activate the Python environment, then from the repository folder:

```bash
mkdir -p /scratch/$USER/showerlm_logs
condor_submit condor/generate_a40.sub                    # 4 jobs x 256 showers on A40 slots
condor_submit condor/generate_1080.sub                   # 8 jobs x 128 showers on GTX 1080 slots
condor_submit condor/generate_cpu.sub jobs=20 n=2        # CPU slots (slow)
```

Job k writes `<outdir>/job_k` with seed `seed0 + k`. Options on the command line: `jobs=`, `n=`, `seed0=`,
`outdir=` (default `showers_a40`, `showers_1080`, `showers_cpu` in the repository), `logdir=` (default
`/scratch/$USER/showerlm_logs`; condor logs may not be written to /data/user), and `cpus=` for the CPU file.

## Differences from the 12 ns generator

[showerlm_generator](https://github.com/qwang552/showerlm_generator) generates the first 12 ns of a shower in a
4 m box. This generator covers the whole shower:

| | 12 ns generator | this generator |
|---|---|---|
| time window | first 12 ns | 80 ns |
| box | 4 m × 4 m × 4 m (192³ voxels) | 20 m × 20 m × 20 m (960³ voxels) |
| voxel | 2.083 cm | 2.083 cm |
| light per shower | about 2 × 10⁷ photons | about 2.5 × 10⁸ photons |
| time per shower | about 1 min (GTX 1080) | about 10 s (A40), 25 s (GTX 1080) |

The output keys are the same, so code written for the 12 ns showers reads these too; only the box and the grid size
(960 instead of 192) change.

## Output

`showers/sample_XXXXX.npz`, one file per shower:

| key | shape | meaning |
|---|---|---|
| `ijk` | (N, 3) | voxel indices 0–959 in x, y, z; x, y ∈ [−10, 10] m, z ∈ [30, 50] m; the primary enters at (0, 0, 50 m) moving down |
| `q` | (N,) | photons in the voxel |
| `off` | (N, 3) | photon-weighted position inside the voxel, in voxel units (−0.5 to 0.5) |
| `m` | (N, 3) | photon-weighted direction of the emitting particles; its length measures how aligned they are |
| `t` | (N,) | photon-weighted time minus the time of a plane front moving down at the speed of light (ns) |
| `cnt` | (N,) | number of emission records |

The keys are the same as in the cache events (`showerlm.common.load_event`). The folder also holds
`tokens_XXXXX.npz` (the generated strokes: `tok_skel`, `tok_det`, and both merged in `tok`), `plans_XXXXX.npz`
and `place_XXXXX.npz` (the plans drawn for the shower and the placement record), `run.json`, `timing.csv` (one row per
shower), `summary.json`, and three plan checks (`gcheck`, `zcheck`, `tcheck`: what the plan asked for and what the
shower got).

## Model

A shower is described as a sequence of straight light strokes: pieces of particle tracks up to 67 cm long, or single
voxels. Each stroke has discrete attributes for time, position, direction, length, photons and a few others.

1. **Plan.** A training shower is drawn at random from `priors/`; its summary is the plan of the new shower:
   9 global features (depth and radius quantiles, light-arrival times, number of long tracks), the light in 12 depth
   slices (0–20 m) and the light in 7 lateness slices (0–8 ns behind the light front). Only these numbers are
   copied, not the shower.
2. **Skeleton.** A transformer generates the long tracks (40 cm or longer), about 2,500 strokes, one at a time,
   conditioned on the plan. Guards keep it on the plan: it may not stop before 70% of the planned light, a new track
   that would put too much light late is redrawn, and the skeleton stops at a new track that starts at 80 ns or later.
3. **Detail.** A second transformer generates the remaining, shorter strokes (about 35,000), grouped by the skeleton
   stroke they belong to. All groups advance together (lockstep). Their total light is set from the plan: the detail
   light of the plan shower, corrected by the fitted slope (`priors/target_fit.json`) for how much brighter or darker
   the new skeleton came out. A placement rule redraws a stroke whose head lands on a voxel that is already lit
   by a track.
4. **Render.** The strokes are drawn into the voxel grid (rvC: along the stroke path, light only after the light cone
   from the injection point).

| part | network | parameters | trained on |
|---|---|---|---|
| skeleton | 6 layers, d 384, 6 heads; attribute decoder 2 layers, d 192 | 22.5 M | 11,018 CORSIKA 8 showers, 8,000 steps (`ft063tz_skel_1080`) |
| detail | 6 layers, d 384, 6 heads; attribute decoder 2 layers, d 192 | 22.2 M | the same showers, 20,000 steps (`ft062ot_det_a40`) |

The checkpoints contain the averaged (EMA) weights in float16.

### Sampler settings

`run.json` holds them; `generate.py` passes each one to `showerlm.sample06`:

| setting | value | meaning |
|---|---|---|
| `skel_min_plan` | 0.7 | the skeleton may not end before it has 70% of the planned skeleton light |
| `skel_late_guard` | 0.05 | redraw a new track that would put too much light more than 8 ns behind the light front |
| `skel_late_jump` | 0.5 | redraw a new track whose start time forces more than half of a depth slice's light late |
| `skel_cap_stop` | 80 | stop the skeleton at a new track that starts at 80 ns or later |
| `target_total`, `target_scale` | prior, nq | detail light and number of strokes set from the plan shower (slope from `target_fit.json`) |
| `place`, `place_occ`, `place_w_occ`, `place_tries` | xu, path, 0.95, 16 | redraw a short stroke whose head lands on a voxel already crossed by a track |
| `sampler` | lockstep | all groups of a shower (and several showers) advance together |
| render | raster2 path, w_min 0.15, causal | draw along each stroke; no light before the light cone from the injection point |

### How good it is

The 0.6.4c evaluation compares 256 generated showers with the CORSIKA 8 test showers on 27 checks (23 numbers and 4
profiles, voxels drawn with rvC). It passes 23 of 27; the true tokens drawn the same way pass 24, which is the
ceiling of this description. The four that fail:

- median photons per lit voxel: +7.6%
- lit voxels within 0.2 m of the axis: −17.1%
- median radius of the lit voxels: +8.7%
- the log10 histogram of photons per voxel

A larger sample of 1,024 showers on the A40 gives the same 23 of 27; 1,024 on a GTX 1080 give 22 of 27, the extra
failure (spread of the time of 90% of the light) coming from a single shower.

Known shortcomings:

- too few lit voxels near the axis (−17% within 0.2 m) and a median voxel radius about 9% too large;
- the light deep near the axis (more than 4 m below the depth of 90% of the light) is about twice the truth, and the deep
  tracks are too aligned with the axis;
- the light after 40 ns is about twice the truth (a small share: 0.04% of the light instead of 0.02%);
- about 1 in 2,000 showers derails: its skeleton stops early (a few hundred strokes), and the shower has about 30% too
  little light and late light out to tens of ns. In `timing.csv` they have an `n_skel` of a few hundred instead of about 2,500.

## Files

| path | what |
|---|---|
| `generate.py` | the one-command generator |
| `run.json` | the fixed sampling settings |
| `showerlm/` | the ShowerLM 0.6.4c package (sampler, models, tokens, renderer) |
| `trained_model/skeleton`, `trained_model/detail` | checkpoints (`best.pt`) and their training configs |
| `priors/gfeat`, `priors/zplan`, `priors/tplan` | plan tables of the training showers |
| `priors/target_rows.npz` | skeleton light, detail light and detail strokes of each training shower |
| `priors/target_fit.json` | the slope of detail light on skeleton light |
| `priors/vocab.json` | the token vocabulary |
| `condor/` | HTCondor submit files (A40, GTX 1080, CPU) |
| `visualize.ipynb` | 2D projections, time profiles and a 3D view of generated showers |
