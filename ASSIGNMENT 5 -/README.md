# Assignment 5 – CO5: Agents, Intelligent Systems and Real-Time AI

This folder contains the report, modular Python demonstrations, test inputs, diagrams and captured console-output images.

## Important academic note
The drone, swarm, expert-system, RDD fallback and chatbot demonstrations are reproducible local simulations. The edge-model benchmark in this package uses a deterministic **synthetic intrusion-like dataset** so that the project runs without downloading data or training a real production model. It is labelled as a demonstration and is **not** presented as a measured result on UNSW-NB15 or another public dataset. For the final assessed B3 submission, download the cited public dataset, follow its licence/terms, use a leakage-safe split, and rerun the benchmark script with real data.

## Requirements
- Python 3.10+
- Standard library for the main demonstrations
- Optional: `pyspark` for the genuine RDD execution path (`pip install pyspark`)
- Optional: `matplotlib` for regenerating charts

## Run
From this folder:
```bash
python src/run_all.py
python src/drone_sim.py
python src/expert_system.py
python src/edge_demo.py
python src/chatbot.py
```
For PySpark:
```bash
pip install pyspark
python src/rdd_pipeline.py
```
If PySpark is unavailable, `rdd_pipeline.py` reports the dependency requirement; the report also includes precise RDD pseudocode.

## Structure
- `Assignment_5_CO5_Report.docx` – formatted report (Times New Roman; justified; 16/14/12 pt; 1.5 spacing)
- `src/` – source code
- `tests/` – deterministic test inputs
- `outputs/` – captured sample logs and metric tables
- `diagrams/` – architecture and workflow diagrams
- `screenshots/` – console/chart evidence images
