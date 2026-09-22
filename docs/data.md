# Data

Pilot disclaimer: FieldLens is a research pilot. Results are trends from small runs on a data subset, not benchmark numbers.

## Source

- Bucket: `intelinair-data-releases`
- Prefix: `agriculture-vision/cvpr_challenge_2021/supervised/`
- Citation: Chiu et al., "Agriculture-Vision: A Large Aerial Image Database for Agricultural Pattern Analysis," CVPR 2020.

## Confirmed S3 layout (Phase 2)

Inspected with unsigned HTTPS ListObjectsV2 (and `inspect_bucket.py`).

Under `supervised/` there are **not** loose tile files. There is one archive:

| Key | Size |
|-----|------|
| `agriculture-vision/cvpr_challenge_2021/supervised/Agriculture-Vision-2021.tar.gz` | 21,033,702,578 bytes (~19.59 GB) |

Also present one level up under `cvpr_challenge_2021/`:

- `Agriculture-Vision Dataset Terms of Use.pdf`
- paper PDF

## Archive internal layout

Confirmed from Hugging Face dataset viewer paths for the same archive:

```
Agriculture-Vision-2021/
  train|val|test/
    images/rgb/<tile_id>.jpg
    images/nir/<tile_id>.jpg   (extension confirm after extract)
    boundaries/<tile_id>.png
    masks/<tile_id>.png
    labels/<class>/<tile_id>.png   (train/val; not on official test)
```

Tile names look like `FIELDID_x1-y1-x2-y2`. Field id is the substring before the first underscore (confirmed on sample names such as `XF6FPLB2I_6774-3675-7286-4187`).

## Classes (2021 supervised)

background (derived), double_plant, drydown, endrow, nutrient_deficiency, planter_skip, water, waterway, weed_cluster.

Each anomaly is a separate binary mask (labels may overlap).

Valid pixels: inside **boundary AND mask**. Invalid pixels are never trained or scored.

## Our split method

Default settings: `training/configs/data.yaml` (shared classes/colours) plus a scale profile.

| Profile | train | val | test | Notes |
|---------|------:|----:|-----:|-------|
| `configs/profiles/pilot_v2.yaml` | 1500 | 400 | 800 | Recommended. `max_tiles_per_field` 10/10/5. Plan used 202/60/188 fields. |
| `configs/profiles/pilot.yaml` | 1500 | 400 | 800 | v1. Few fields (31/15/18). Keep for comparison only. |
| `configs/profiles/full.yaml` | 56944 | 18334 | 19708 | Official sizes; untested placeholders |

### Why pilot_v2 replaced pilot (v1) as the working split

pilot v1 filled tile budgets from whole fields without a per field tile cap. That used only 31 train, 15 val and 18 test fields. Class mixes then differed strongly (example from `data/fieldlens/pilot/stats.json`: drydown tile counts 80 / 1 / 566). Val and test could rank runs differently for that reason. Keep v1 numbers as a finding under `runs/pilot/`, not as headline results.

pilot_v2 keeps 1500/400/800 and seed 42, adds `max_tiles_per_field`, and still separates fields across splits.

```bash
python scripts/download_subset.py --profile pilot_v2 --mode plan
python scripts/download_subset.py --profile pilot_v2 --mode extract
python scripts/compute_stats.py --profile pilot_v2
```

Data for a profile lives under `data/fieldlens/<profile>/`.
Extract uses one sequential pass over the `.tar.gz`. Tile ids are cached at
`data/raw/Agriculture-Vision-2021.tar.gz.tile_ids.json` after the first scan.

1. Group official **train** tiles by field id. Shuffle field ids with seed 42.
2. Fill **our train** from whole fields, then **our val** from remaining fields.
3. From official **val**, same method for **our test**.
4. Code asserts no field id appears in two of our splits; fails loudly on overlap.
5. Write `data/fieldlens/split.csv` with columns: `tile_id,source_split,our_split,field_id`.

Because the remote object is a tar.gz, download the archive once, then run `download_subset.py --mode extract` to copy only our subset.

A previous large plan (about 5456 rows) was saved as `data/fieldlens/split_large_planned.csv` before switching to the small default.

## Singlelabel rule (Run 1)

For each valid pixel:

- No anomaly -> 0 (background)
- Exactly one anomaly -> that class id (1..8)
- Two or more anomalies -> the **rarest** of them by train valid-pixel frequency from `data/fieldlens/stats.json`
- Invalid -> 255 (ignored)

## Stats

`training/scripts/compute_stats.py` writes `data/fieldlens/stats.json`: tiles/fields per split, per-class valid pixels and tile counts, overlap pixel counts, NIR mean/std on our train only.

Stats summary: **pending** until AK runs download + compute_stats and pastes output.

## Disk

Need roughly 20 GB for the archive plus subset extract space (several GB). Check with `df -h`.
