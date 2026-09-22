# Report mapping

Pilot disclaimer: FieldLens is a research pilot. Results are trends from small runs on a data subset, not benchmark numbers.

| Report section | Supporting files / figures |
|----------------|----------------------------|
| Abstract | `README.md` summary; `dashboard/public/data/results.json` metrics (after runs); pilot disclaimer |
| Introduction | `README.md`; `docs/data.md`; dashboard `/` overview |
| Literature Review | Citations in README and `/method` (Agriculture-Vision, SegFormer, AgriFusion) |
| Problem Formulation | Research questions in README; multi-label overlap discussion in `docs/data.md` and `docs/evaluation.md` |
| Dataset Preparation | `docs/data.md`; `training/scripts/download_subset.py`; `data/fieldlens/split.csv`; `stats.json`; notebook data preview / class distribution |
| Model Architecture | `training/fieldlens/models.py`; dashboard `/method` SVG; notebook architecture notes |
| Code Implementation: data loading | `training/fieldlens/dataset.py`; `transforms_drone.py`; tests |
| Code Implementation: model definition | `training/fieldlens/models.py`; `tests/test_models.py` |
| Code Implementation: training and validation | `training/train.py`; `configs/run*.yaml`; `runs/*/log.csv` |
| Code Implementation: prediction and visualization | `training/evaluate.py`; `export.py`; dashboard `/explorer`, `/compare`; notebook sample predictions |
| Model Evaluation | `docs/evaluation.md`; `runs/*/eval/`; dashboard `/results`; robustness and efficiency JSON |
| Conclusion | Results table in README (from real `results.json` only); reflection from eval docs |
| PDF: project definition | README + agenda Phase 0-1 |
| PDF: design and implementation | method page + training package |
| PDF: execution and testing | `docs/training.md`, pytest, smoke/benchmark logs |
| PDF: evaluation and reflection | `docs/evaluation.md`, results page, notebook figures |
