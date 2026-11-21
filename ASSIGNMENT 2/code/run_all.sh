#!/bin/bash
# Regenerates every result and figure of Assignment 2 (Python 3 + matplotlib required).
# Fixed random seeds are used everywhere; operation counts are reproducible, timings vary with the machine.
set -e
cd "$(dirname "$0")"
mkdir -p ../results ../figures
python3 task1_search.py
python3 task2_online.py
python3 task3_alphabeta.py
python3 task4_minimax.py          # about 1 minute
python3 task5_csp.py
python3 task6_tsp.py              # about 1 minute (12-city exhaustive run takes ~40 s)
python3 task7_redesign.py         # about 3 minutes
python3 make_figures.py
