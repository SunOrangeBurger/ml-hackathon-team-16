# Data and model training

The challenge brief schedules Review 1 before lunch and describes a final evaluation. The Review 1 checklists below are proposed team milestones. The brief gives no separate checklist for that checkpoint.

## Review 1 requirements

- [x] Inventory the 80 training and 10 student-testing trips and document file schemas, missing data, timestamp gaps and road connectivity.
- [x] Generate a Folium map for a training trip with raw GPS, road geometry and the supplied reference route.
- [x] Demonstrate candidate generation and label alignment on a training trip. Record baseline candidate coverage and export a scored fixture for the decoder.

See the [Review 1 run instructions and results](review-1-data-and-model.md).

## Final evaluation requirements

- [ ] Compare at least five trained models using trip-grouped cross-validation on the 80 training trips and development comparisons on the 10 student-testing trips.
- [ ] Select a model with measured evidence, run Bayesian optimization using training-trip folds and compare tuned and untuned models on the same partitions and metrics.
- [ ] Export the final model, fitted preprocessing, feature schema and inference configuration. Verify that reloaded artifacts reproduce candidate scores.
- [ ] Supply reproducible experiment logs, tables and plots for the final report. Keep reserved evaluator trips outside development.

## Detailed tasks

1. Parse whitespace-delimited files. Read longitude, latitude and seconds from `.track`, node coordinates from `.nodes`, directed node pairs from `.arcs`, and chronological arc IDs from `.route`. Preserve trip-local row IDs and leading zeros in trip IDs. Validate arc endpoints and reference IDs.
2. Profile file coverage, observation counts, coordinate ranges, timestamp order and gaps, isolated nodes and connected components. Document the handling of invalid rows and repeated observations. Choose a suitable metric projection or geodesic calculation.
3. Generate nearby directed candidates by point-to-segment distance. Compute features available at inference, such as distance, heading agreement and elapsed time. Document the radius, candidate cap, ordering and fallbacks. Estimate reference alignment using the ordered reference route and report ambiguous labels and candidate misses separately.
4. Build a common experiment pipeline that fits preprocessing inside each trip-grouped fold. Compare five distinct models, such as logistic regression, k-nearest neighbors, random forest, histogram gradient boosting and an MLP. Log seeds, folds, model settings, timing and probability handling.
5. Record CV and student-test macro F1, observation-level top-1 accuracy, candidate miss rate, applicable log loss and inference time. Include precision, recall and diagnostic plots where applicable. Explain overfitting, limitations and the selection objective using measured results.
6. Define the Bayesian search space, objective, trial budget and seed. Save trial history and runtime, refit the selected configuration and compare it with its untuned version. Produce comparison charts and retain all artifacts needed to rerun the experiments.
7. Expose a scoring function that accepts ordered observations and a directed graph and returns one candidate list per observation index. Each candidate carries a trip-local arc ID, its projected position along the arc, and a finite probability in [0, 1]. Document score semantics and use the same feature order and transformations in training and inference.

## Handoff

Provide a real training-trip fixture and a small synthetic fixture to [route reconstruction](02-route-reconstruction.md). Give [FastAPI](03-fastapi-backend.md) the fitted artifacts, schema version and a parity check for offline and service inference.
