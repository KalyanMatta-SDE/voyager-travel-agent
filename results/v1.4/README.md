# v1.4 Replay Benchmark Results

## Summary

v1.4 introduces deterministic replay for LLM-backed agents (intent parser and experience agent). Intent and experiences are now recorded in fixtures during capture, enabling bit-identical replay runs.

**Key Achievement**: Two consecutive replay runs produced identical results on all 24 query pairs (12 queries × 2 modes).

## What Changed from v1.3

- **Intent parser**: Now records intent in `fixtures/live_inventory/*_intent.json`
- **Experience agent**: Now records experience list in `fixtures/live_inventory/*_experiences.json`
- **Replay anchor**: Pinned to timezone-independent anchor date for stable fixture timestamps
- **Error handling**: Research-round errors now recorded in state for proper replay

## Results

The numbers differ from v1.3 because the experience list is now one recorded sample instead of regenerated each run:

- **unsatisfiable_constraint_rate_pct**: 64% → 67%
- **baseline conflicts**: 1.28 → 1.25 (all rounds)
- **location_mismatch detections**: 17 → 16 total, baseline 9 → 8

All other aggregate metrics remained within tolerance. Per-query conflict counts are identical between runs, with q0 (Greece) no longer requiring tolerance - the timing_inefficiency variance that appeared/disappeared in v1.3 is now fixed by recording intent.

## Reproducibility

```bash
python3 scripts/benchmark_queries.py --mode compare --inventory replay --query-count 12
```

Run1 and Run2 produced byte-identical `conflicts_by_round.csv` and `conflict_lifecycle.csv` files, and numerically identical `run_summary.json` values (24 query pairs, mean_final_conflicts=1.17, resolution_rate=10%).

## Files

- `run1/`: First replay run (reference)
- Expected values updated in `benchmarks/expected_replay.json`
