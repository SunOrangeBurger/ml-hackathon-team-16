# Probabilistic map-matching HMM

Set PATHSNAP_DECODER=hmm to select the geometric HMM. The default weighted decoder remains available with PATHSNAP_DECODER=weighted.

The state at each time is a candidate directed arc and projected position. The initial distribution is uniform over that observation's candidates. GPS likelihoods follow an isotropic Gaussian in projected metres. PATHSNAP_HMM_SIGMA_M controls its standard deviation, defaulting to 5 m based on student-test selection.

For each pair of successive observations, feasible directed-network transitions receive the existing displacement, excess-distance and reverse-edge log weights. A stable log-sum-exp normalization converts each source row into an outgoing probability distribution. Forbidden transitions have zero probability. A row with no reachable successor remains zero, permitting the existing section-restart handling.

This is a time-varying map-matching HMM. Its candidate sets and transition distributions depend on the observed interval and GPS displacement. It is not a stationary, jointly trained generative trajectory model. The Gaussian emissions use geometry directly. Classifier probabilities do not enter HMM scores.

Viterbi runs in log space and selects the maximum-probability state sequence for each connected observation section. Empty candidates, nonpositive timestamps, long gaps and wholly infeasible transitions split the sequence.

The implementation shares graph search, route construction and backtracking with the weighted decoder. The models endpoint returns no classifier choices in HMM mode, and the UI labels it Geometric HMM. Recorded classifier catalog accuracy is hidden in this mode.

PowerShell startup from the repository root

```powershell
$env:PATHSNAP_DECODER = 'hmm'
$env:PATHSNAP_HMM_SIGMA_M = '5'
$env:PATHSNAP_MODE = 'evaluator'
$env:PATHSNAP_EVALUATOR_MANIFEST = (Resolve-Path 'output/evaluator-manifest.json').Path
python -m uvicorn review_api:app --host 127.0.0.1 --port 8002
```

Point the evaluator's API address at http://127.0.0.1:8002. Use the same dependency environment and dataset layout as the primary service. The shared service still validates its configured classifier artifact and uses that artifact's candidate radius/cap.
