# Input SNR of the Sonoro ring attractor

Analysis of experiment A1 of 16 November 2026 (ten runs; A1_02 is left out). It works
backwards from the `hext` rows of `ringStats/`, which are exactly what the ring attractor
received in each control cycle, to the signal-to-noise ratio of the sound input in the
terms of the ICRA 2027 paper. The full account is in `report/sonoro_input_snr.pdf`.

## Result

Per-sample SNR in the paper's convention, `D = Δq/η = √2 Δq/σ_o`, where the sample is
`o = q_L − q_R`, the difference between the lobes of the left and right speakers in the
ring input, in units of the taller lobe. Left speaker at 1. Brackets are 95 % bootstrap
intervals. A positive `D` favours the left speaker.

| Right level | Window | Δq | σ_o (A / B) | D, Route A | D, Route B |
|---|---|---|---|---|---|
| 0.7 | start, cycles 0 to 14 | 0.12 to 0.13 | 0.051 / 0.048 | 3.4 [2.7, 4.9] | 3.8 [2.9, 5.8] |
| 0.6 | start, cycles 0 to 14 | 0.19 | 0.042 / 0.042 | 6.4 [4.4, 10.1] | 6.3 [4.0, 11.9] |
| 0.7, four runs that turned right | last 8 cycles before the turn | −0.12 to −0.13 | 0.035 / 0.020 | −5.3 [−10.2, −3.9] | −8.7 [−18.8, −5.7] |
| 0.6 | last 8 cycles before the turn | 0.18 | 0.027 / 0.011 | 9.2 [5.9, 16.3] | 23.0 [15.9, 50.0] |

The ICRA grid is D = 0.25, 0.5 and 1. The signal depends on where the robot is: with the
right speaker at 0.7 it changes sign seven to eleven cycles before the turn, and in all ten
runs the robot turned to the lobe that was taller just before the turn.

## The two routes

Route A regresses the log ratio of the two lobes on the measured geometry (Qualisys) and
takes the residual as noise. Route B uses `ringStats/` only: the robot moves 13 mm per
cycle, so half the variance of the differences between consecutive cycles is the noise
variance. Route A can only overstate the noise, Route B can only understate it.

## Running it

From the repository root:

```
python snr_analysis/sonoro_snr.py       # about four minutes; writes snr_analysis/out/
python snr_analysis/make_figures.py     # writes snr_analysis/fig/
```

`sonoro_snr.py` needs numpy and pandas (scipy only for one normality test), and
`make_figures.py` needs matplotlib. Options: `--boot` (bootstrap replicates, default 2000),
`--seeds` (replays of each run with other random seeds, default 200), `--root`, `--out`.
The run list, the speaker levels and the reported outcomes are set at the top of
`sonoro_snr.py`.

## Files

| File | Content |
|---|---|
| `out/cycles.csv` | one row per control cycle: lobes, sample, geometry |
| `out/precommit_samples.csv` | the samples before the turn, with fit and residual of Route A |
| `out/snr_table.csv` | signal, noise and SNR per condition and window, both routes, with intervals |
| `out/snr_along_approach.csv` | the same in blocks of seven cycles counted back from the turn |
| `out/runs.csv`, `out/decisions.csv`, `out/seed_replay.csv` | per-run checks, what decided each run, replays with other seeds |
| `out/route_a_parameters.csv` | the fitted regression of Route A |
| `out/window_sensitivity.csv`, `out/noise_acf.csv`, `out/noise_phase_locking.csv`, `out/summary.json` | checks and noise diagnostics |
| `out/console_log.txt` | printout of the run that produced these files |
| `fig/` | the five figures of the report, PDF and PNG, at IEEE column widths |
| `report/` | the report, PDF and LaTeX source |

## Things to know

The controller rescales the input so that the taller lobe is always 0.19. Only the ratio of
the two lobes survives, so the acoustic SNR in decibels cannot be recovered from these logs.

`thymioScripts/ring_circle_target.py` in this repository is not exactly the file that ran on
the robot (it names its output differently and takes no run ID). Replaying its ring on the
logged inputs reproduces every logged spin, so the ring and its parameters are the same;
the audio front end cannot be checked from the logs.
