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

## 18 November 2026 — Integrated Review 1 backup

| Area | Decision and reason | Effect and follow-up |
| --- | --- | --- |
| API fixture | The development service runs the Viterbi decoder against recorded candidate scores when a trip is selected. | The UI exercises the actual interface and route reconstruction. Model-based scoring remains a final-evaluation task. |
| UI visualization | The evaluator renders the returned coordinates as a local plan view, so the demo works without external map tiles. | The separately generated Folium audit map supplies the geographic basemap, candidate roads, and development reference overlay. |
| Laptop verification | Verified the API and UI as separate HTTP services on one machine. Edge loaded both fixture trips, rendered both layers, and recovered from a simulated 503 using Retry. The API logged successful requests and an unknown-trip 404. | Repeat the browser request from a second physical laptop and capture the corresponding API log during the review. |

On trip `00000001`, the decoder assigned all 1,315 observations, split the route into two sections after one impossible transition, and agreed with 97.34% of the provisional aligned labels. This measurement uses one training fixture and serves as an integration check.

## 18 November 2026 — Trained pipeline on the backup branch

- Use provisional ordered alignment and exclude observations more than 50 m from the aligned reference. The exclusion disproportionately affects trip `00000054`.
- Sample per trip for comparison and Bayesian search, then refit the selected configuration on all eligible training observations. The search retained the starting histogram gradient boosting configuration.
- Keep the existing Viterbi transition settings for the measured continuity improvement. Pointwise agreement declined on student test, so the results document both outcomes.
- Support organizer-provided trip directories through a server-side manifest. Reserve the physical two-laptop evaluator demonstration for when the official data is available.

See [development results](development-results.md).

## 18 November 2026 — Organizer holdout and decoder correction

- Keep `evaluator_test` completely outside training and configuration selection. It contains ten trips with reference routes and is used once as an organizer holdout.
- Permit up to 30 m of backward projected motion on the same directed arc to absorb GPS jitter. Select transition scale and slack on student-test trips.
- Retain histogram gradient boosting. The CUDA listwise ranker produced only a small, inconsistent grouped-CV improvement.
- Deploy the corrected decoder defaults after they improved student-test pointwise accuracy to 92.21% and organizer-holdout accuracy to 93.39%.
