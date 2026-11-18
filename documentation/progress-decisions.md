# Progress and development decisions

Use this log when implementation changes the order, scope, or approach in a workstream guide. Record the original plan, what changed and why, its effect on downstream work, and any follow-up.

## 18 November 2026 — Workstream 01, first training-trip checkpoint

| Area | Decision and reason | Effect and follow-up |
| --- | --- | --- |
| Dataset exploration | Began with training trip `00000001` to validate file parsing, arc IDs and candidate geometry before running the full inventory. | The other training trips and student-test trips remain to be profiled. This run establishes a working example for the broader audit. |
| Reference alignment | Exported a diagnostic flag that indicates whether a candidate arc appears anywhere in the ordered reference route. The raw route gives traversed arcs and has no per-observation labels. | Keep the flag out of model features and training targets. Implement and inspect order-aware alignment before reporting candidate miss rate, top-1 accuracy or producing labeled examples. |
| Candidate search | Started with a 50 m radius and at most 10 directed candidates per observation, ranked by point-to-segment distance and then arc ID. Distances use a local equirectangular approximation centered on the trip. | These are configurable baseline values. Check the approximation and measure the effect of radius and cap across training trips after validating alignment. |

The first run produced a Folium map, a candidate CSV and a JSON summary for 1,315 GPS observations. All observations had candidates. Of those, 1,313 had at least one candidate that appeared somewhere in the reference route. That count measures route membership, while observation-level coverage requires aligned labels.

Reproduce from the workspace root with:

```powershell
python -m pip install -r requirements.txt
python inspect_trip.py --trip 00000001
```

The outputs are written to `output/trips/00000001/`.

## 18 November 2026 — Workstream 01 Review 1 completed

| Area | Decision and reason | Effect and follow-up |
| --- | --- | --- |
| Candidate coverage | Replaced route-membership counts with per-observation comparisons against a monotonic, distance-based reference alignment. | The metrics apply to provisional labels on one example trip. Check alignment around long gaps and discontinuous references before scaling up. |
| Decoder fixture | Used normalized distance weights to provide scores before training a classifier. | The decoder can start integration now. Replace the scores with outputs from the selected model once training is complete. |
| Dataset audit | Found a 153,064-second GPS gap and 13 reference discontinuities in training. | Use this evidence when defining gap and discontinuity handling for batch labeling and decoding. |

See [Review 1 results and reproduction steps](review-1-data-and-model.md).

## 18 November 2026 — Workstream 02, Review 1 decoder

| Area | Decision and reason | Effect and follow-up |
| --- | --- | --- |
| Candidate interface | Reused the candidate fields already exported by Workstream 01: observation index, trip-local directed arc ID, source-to-destination arc fraction and probability in `[0, 1]`. | This provides a concrete decoder handoff without changing the model workstream. Confirm scorer-owner agreement and parity when the trained scorer is integrated. |
| Implementation scope | Added a standalone, standard-library decoder and a checked-in synthetic graph/score fixture rather than adding a model or API dependency. | Synthetic Review 1 demo and tests run without the unavailable dataset. Run the file-backed command when Workstream 01 artifacts and data are present. |
| Transition policy | Used shortest directed network distance, a 55 m/s hard speed ceiling, a 15 m/s soft transition scale, a 1 m tolerance and a 60-second gap restart for the demonstration. | These defaults make behavior explicit and reproducible, but are not calibrated. Tune using allowed development trips before final evaluation. |
| Discontinuities | Empty observations remain explicitly unmatched; long gaps and impossible transitions split route sections and generate warnings rather than joining them with straight lines. | FastAPI can consume assignments, sections and warning codes. Validate the response shape during backend integration. |

Verification on this checkpoint: the synthetic pointwise winners are `[0, 2]` and the decoded connected assignments are `[0, 1]`; all eight focused standard-library tests pass. A generated real candidate fixture is present under the ignored `output/review-1/` directory, but its referenced graph files are not currently available at the documented `Student folder/Student folder` path, so the real-fixture decode could not be rerun. Trained-scorer parity, tuning and runtime measurements remain follow-ups.
