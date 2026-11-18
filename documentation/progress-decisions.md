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
