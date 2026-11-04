# CSE 620 — Simulated Annealing Micro-Project

Report + experiments for the *Official CSE620 SA Project [Template].docx*.

## Layout

| Path | Purpose |
|------|---------|
| `experiments.py` | Runs all required experiments (Sections 1–3) and writes `results/*.csv` + report figures |
| `report/` | Self-contained LaTeX project — upload this folder to Overleaf |
| `CSE620_SA_Project_Notebook.ipynb` | Original course notebook |
| `.venv/` | Dedicated Python environment (git-ignored) |
| `results/` | Generated tables (CSV), logs and figures |

## Reproducing the numbers

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -c "import experiments as e, os; os.makedirs('results', exist_ok=True); print(e.section1()); print(e.section2()); print(e.section3('linear', 0.95, 4000)); e.cooling_curves()"
```

## Building the report

```bash
cd report
latexmk -pdf main.tex
```

Or open it in Overleaf: **New Project → Upload Project** and upload a zip of
the `report/` folder (it already contains `main.tex` and `figures/`).

## Notes

* The provided sanity-check costs reproduce exactly:
  exponential 2000/0.60 → **3963.2**, linear 2000/0.60 → **2894.4**.
* Wall-clock times depend on the machine; all rows are timed on the same one.
