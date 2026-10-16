# Data for the local R1 revision

The fixed-R1 command `python code/run_all.py local` reads the frozen inputs in `code/results/r1_reference/`. It recomputes the corrected false-alarm and two-stage latency bounds, reproduces the original nominal detection estimates, regenerates the table and figure, and runs the consistency checks. It uses the retained outputs for experiments unaffected by the correction. It does not rerun the real-traffic experiments or acquire traffic.

## Real-traffic pools

The original real-traffic experiments use NumPy `.npy` arrays of per-packet inter-packet delays, measured in milliseconds, extracted from the ISCXTor2016 and ISCXVPN2016 captures. The arrays and raw captures are not included in this package. The original pooled arrays contain 9,214,592 Tor delays and 1,850,729 VPN delays. Dataset references are retained in the manuscript bibliography.

To rerun these experiments with the original pools, set `NFW_TRACE_DIR` to their directory. The scripts use `iscx_tor_ipd.npy`, `iscx_vpn_ipd.npy`, Tor application files named `iscx_tor_{browsing,chat,mail,video,p2p,file_transfer,audio}_ipd.npy`, and VPN application files named `iscx_vpn_{audio,chat,video,mail,file_transfer}_ipd.npy`, as required by each probe. These filenames describe separate arrays, not literal brace-containing filenames. The original extraction computes adjacent-packet timestamp differences within each unidirectional flow. The extraction program is external to this revision package.

The retained `code/results/probeA*.json` files supply the reported real-traffic measurements. Regenerating the figure from these files does not remeasure the traffic. Header selection and false-hit observations on the tested pools remain empirical findings; they do not establish the independent-background certificate for dependent traffic.

## Synthetic experiments

Synthetic probes generate their inputs with the fixed NumPy seeds recorded in their scripts. The nominal detection simulation and its seed are unchanged in this revision. Numerical reproduction requires the dependencies in `code/requirements.txt`; identical Monte Carlo outputs should be assessed using the recorded NumPy version in the reproduction report.
