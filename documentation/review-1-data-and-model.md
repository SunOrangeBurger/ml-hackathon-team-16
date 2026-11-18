# Workstream 01 Review 1 results

Run from the project root:

```powershell
python -m pip install -r requirements.txt
python review_one.py
```

`--data-dir` points to the directory containing `train` and `student_test`. Results default to `output/review-1/`.

## Dataset inventory

Files contain whitespace-separated rows without headers. IDs are zero-based and local to each trip.

| File | Columns |
| --- | --- |
| `.track` | Longitude, latitude, timestamp in seconds |
| `.nodes` | Longitude, latitude. The row number is the node ID. |
| `.arcs` | Source node ID, destination node ID. The row number is the directed arc ID. |
| `.route` | Arc ID for each reference route step, in traversal order |

The audit found all 360 expected files. All tables parsed with finite values and valid coordinate bounds, arc endpoints and route IDs. Neither partition had repeated or decreasing timestamps or any isolated nodes.

| Measure | Train | Student test |
| --- | ---: | ---: |
| Trips | 80 | 10 |
| GPS observations | 208,417 | 19,160 |
| Observations per trip | 361–19,313 | 262–3,958 |
| Timestamp steps above 1 s | 7,286 | 264 |
| Largest gap | 153,064 s in `00000090` | 210 s in `00000065` |
| Trips with multiple weakly connected components | 79 | 10 |
| Weakly connected components per trip | 1–421 | 5–100 |
| Strongly connected components per trip | 2–1,942 | 6–415 |
| Reference arc discontinuities | 13 across 10 trips | 1 across 1 trip |

`dataset_inventory.csv` contains file sizes, row counts, coordinate bounds, timestamp diagnostics and directed graph statistics for each trip. The large gap and route discontinuities need explicit handling before applying the labeling method across the training partition.

## Training trip 00000001

Point-to-segment distances use a local metric approximation. For each GPS observation, the baseline retains up to 10 directed arcs within 50 m, sorted by distance and then arc ID. A separate dynamic program aligns GPS points to reference route positions by minimizing squared distance while requiring the positions to stay in order. This creates provisional labels even when the observation could plausibly belong to another road. Alignment uses the complete reference route independently of the candidate cap.

| Metric | Result |
| --- | ---: |
| GPS observations | 1,315 |
| Observations with candidates | 1,315 |
| Aligned arc retained | 1,313 / 1,315 (99.85%) |
| Candidate miss rate against aligned arcs | 2 / 1,315 (0.15%) |
| Nearest candidate agrees with aligned arc | 1,048 / 1,315 (79.70%) |

Both misses had aligned distances below 50 m. Observations 1099 and 1140 lost their aligned arcs at the top-10 cutoff.

## Review artifacts and decoder handoff

- `00000001/map.html` overlays raw GPS, nearby candidate roads and the supplied reference route.
- `00000001/candidates.csv` contains distance ranks and projected arc fractions. Its route-membership flag is a development diagnostic.
- `00000001/aligned_reference.csv` and `00000001/baseline_metrics.json` contain provisional labels and aggregate diagnostics.
- `00000001/decoder_fixture.json` contains observations and candidate scores. Resolve graph file paths relative to `--data-dir`. Coordinates are longitude then latitude, and the arc fraction runs from the source node at 0 to the destination node at 1. Scores are normalized Gaussian distance weights with a 15 m scale.
- `synthetic_decoder_fixture.json` embeds a directed graph with the same candidate fields. The independent highest-score sequence is `[0, 2]`. The connected sequence is `[0, 1]`.

Keep aligned reference labels and route-membership flags out of decoder inputs and production inference. Model training and calibration are final-evaluation tasks.
