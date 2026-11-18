# HMM and Viterbi route reconstruction

The challenge brief schedules Review 1 before lunch and describes a final evaluation. The Review 1 items below are proposed team milestones; the brief does not define a separate Review 1 checklist.

## Review 1 deliverables

- [x] Define the scorer-to-decoder candidate interface and decode fixed synthetic scores.
- [x] Show that independent pointwise maxima can select a disconnected route while Viterbi selects a feasible directed sequence.
- [x] Define and test how empty candidates, timestamp gaps, duplicate timestamps, invalid inputs and impossible transitions are reported.

Run the self-contained demo and tests from the project root:

```powershell
python route_reconstruction.py
python -m unittest discover -s tests -p test_route_reconstruction.py -v
```

The fixed fixture is [synthetic_decoder_fixture.json](../fixtures/route-reconstruction/synthetic_decoder_fixture.json); its decoder settings are in [decoder_config.json](../fixtures/route-reconstruction/decoder_config.json). In the fixture, independent highest scores select `[0, 2]`, where arc 2 is disconnected from arc 0. The decoder instead selects `[0, 1]`, a connected directed route.

To decode the real Workstream 01 fixture after generating it with `review_one.py`:

```powershell
python route_reconstruction.py `
  --fixture output/review-1/00000001/decoder_fixture.json `
  --data-dir "Student folder/Student folder" `
  --output output/route-reconstruction/00000001.json
```

The command reads file-backed graphs relative to `--data-dir`. It writes assignments aligned to fixture observation IDs, ordered route sections, GeoJSON-style longitude/latitude coordinates (but not a GeoJSON Feature), configuration used, and structured warnings. It does not read aligned reference labels or route-membership diagnostics.

## Code explainer and contract

The standalone implementation is [route_reconstruction.py](../route_reconstruction.py). Its input contains one observation object per ordered GPS observation:

```json
{
  "observation_index": 0,
  "timestamp_s": 1.0,
  "candidates": [
    {"arc_id": 12, "arc_fraction": 0.4, "probability": 0.8}
  ]
}
```

`arc_id` is the row index of a directed arc in the trip-local graph. `arc_fraction` is a position from the source node (0) to destination node (1). `probability` is the scorer's finite candidate probability in `[0, 1]`; candidate probabilities need not sum to one. The decoder validates unique increasing observation indices, nondecreasing finite timestamps, valid arc IDs and positions, and unique states within each observation. Invalid inputs fail explicitly rather than producing a partial result.

The HMM hidden state is the candidate's directed arc and fraction. Its emission log-score is `log(max(probability, probability_floor))`. The floor prevents zero-probability candidates from introducing negative infinity; a candidate with zero probability remains strongly disfavored. Candidate states are ordered by arc ID and fraction, and equal-scoring paths resolve to the first such state and predecessor, making ties deterministic.

For each candidate pair, the graph computes the shortest legal directed network distance between projected positions: forward movement on the same arc is direct; otherwise travel continues to the source arc's destination, follows directed intermediate arcs, and ends at the target arc's fraction. Repeated arc traversals and loops are permitted when the directed graph provides a path. Shortest node-to-node results are cached (up to 8,192 pairs). Arc lengths use a local equirectangular metric projection, appropriate for the local road graphs in the fixtures.

For candidates `i` and `j` in consecutive observations, elapsed time is `dt` and the transition is impossible when there is no directed path or its distance exceeds `max_speed_mps * dt + distance_tolerance_m`. Otherwise the log-transition score is:

```text
-0.5 * (network_distance / (transition_speed_scale_mps * max(dt, 1 second)))^2
```

Viterbi maximizes the sum of log emissions and log transitions:

```text
V[t, j] = emission[t, j] + max_i(V[t-1, i] + transition[i, j])
```

Backpointers recover observation assignments. The chosen network paths between assignments are separately expanded into arc traversals and connected route geometry; no straight line is drawn between disconnected sections.

### Empty candidates and discontinuities

- An empty candidate list produces an assignment with null `arc_id`, `arc_fraction` and `probability`, plus an `empty_candidates` warning. It terminates the current route section.
- A gap strictly greater than `max_gap_s` terminates the section before the next observation and starts a new one, with a `timestamp_gap` warning. Equality to the threshold does not restart.
- Duplicate timestamps are permitted. A transition is possible only when its network distance is within the configured distance tolerance; otherwise it is treated as an impossible transition.
- If all transitions into an observation are impossible (including a one-way or disconnected movement, or a movement above the speed limit), the decoder emits `no_legal_transition` and restarts from that observation as a new route section. It preserves the observation's best candidate rather than fabricating a connector.
- Warnings are objects with a stable `code`, relevant observation index, and explanatory message. Malformed input raises an error; the CLI prints the failure and exits nonzero.

## Decoder configuration

| Setting | Review 1 default | Meaning |
| --- | ---: | --- |
| `probability_floor` | `1e-12` | Minimum emission probability before taking the logarithm |
| `max_gap_s` | `60` | Larger observation intervals restart decoding |
| `max_speed_mps` | `55` | Hard upper bound used to reject implausible transitions |
| `transition_speed_scale_mps` | `15` | Scale for the soft transition penalty |
| `distance_tolerance_m` | `1` | Allows small projection/rounding error in speed checks and zero-time steps |

These are Review 1 demonstration defaults, not trained or calibrated settings. Configuration is included in every decoder result. The command accepts a replacement JSON file with `--config`.

## Dependencies and run instructions

The decoder and its tests use only the Python standard library; no package installation is required to run the synthetic demo. From the repository root, use Python 3.8 or newer:

```powershell
python route_reconstruction.py
python -m unittest discover -s tests -p test_route_reconstruction.py -v
```

The real-fixture command additionally needs an existing fixture and its graph files. Generating Workstream 01 fixtures uses the project dependencies from [requirements.txt](../requirements.txt); install them and create the fixture first:

```powershell
python -m pip install -r requirements.txt
python review_one.py --data-dir "Student folder/Student folder"
```

## Final evaluation requirements

- [ ] Decode the selected model's scores into per-observation assignments and a directed route, with gaps clearly marked.
- [ ] Compare classifier-only and classifier-plus-HMM on identical development trips. Report pointwise accuracy, continuity, sequence performance and examples where decoding helps or hurts.
- [ ] Tune settings on permitted data; export the selected decoder configuration and document hidden states, emissions, transitions, recurrence and fallback behavior.
- [ ] Save representative failure maps and timings. Keep aligned references and reference-derived labels out of decoder inputs and production inference.

### Detailed final-evaluation tasks

1. Connect the selected scorer through the candidate interface above and verify observation and arc ID parity.
2. Tune distance, gap and transition settings using permitted training and student-testing data; do not treat the Review 1 defaults as measured optima.
3. Compare both variants using identical aligned references and denominators. Report pointwise accuracy, directed continuity, a defined sequence metric such as arc edit distance, and cases where decoding helps or hurts.
4. Verify one-way roads, duplicate timestamps, repeated arcs and disconnected graphs on real fixtures. Record decoder runtime and representative failure maps.

## Handoff

Return assignments aligned to GPS indices, ordered route sections and gap warnings to [FastAPI](03-fastapi-backend.md). Use the fixture and score semantics from [data and model training](01-data-and-model-training.md).
