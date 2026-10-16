# Detection-Theoretic Sync-Header Design for IPD-QIM Timing Watermarks under Insertion–Deletion Channels

Code and results for the paper (IEEE Signal Processing Letters). Every number the
manuscript reports — the seven designs of Table I, the certified finite-window
false-alarm bounds, the nominal detection estimates, the real-traffic measurements
and Fig. 1 — is regenerated end to end by one command:

```
python3 code/run_all.py local
```

## Contents

- `code/` — library (`design_v2.py`), one script per experiment (`experiments/`),
  table/figure/consistency reports (`report/`), and the committed result JSONs
  (`results/`). See `code/README.md` for the stage table and function reference.
- `manuscript/` — LaTeX sources; `code/report/make_tables.py` and `make_figs.py`
  write Table I and Fig. 1 there.
- `DATA.md` — format and availability of the real-traffic IPD pools.

## Requirements

Python >= 3.10 with `numpy` and `matplotlib` (`pip install -r code/requirements.txt`).
No SciPy: all binomial tails are exact sums computed in `design_v2.py`. Seeds are
fixed; on the recorded NumPy version the committed JSONs are reproduced exactly.

## Real-traffic data

The Tor and VPN IPD pools are extracted from the ISCXTor2016 and ISCXVPN2016
captures (references [15], [16] of the paper) and are not redistributed here.
Set `NFW_TRACE_DIR` to rerun the A-series stages from them; every other stage
runs out of the box and the committed `results/probeA*.json` carry the reported
measurements. See `DATA.md`.

## License

MIT for the code (`code/LICENSE`). The datasets remain under their own terms.
