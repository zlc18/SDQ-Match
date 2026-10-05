# SDQ-Match benchmark report (generated)

All numbers: paired common-random-number evaluation, seeds 200-229 (scale runs: 200-209). Gains are relative QASS gains of SDQ-Match; p values are two-sided Wilcoxon signed-rank, Holm-corrected over the baselines of a scenario.

| scenario | SDQ-Match | strongest tier-1 (published, as specified) | gain | p_Holm | strongest tier-2 (forecast-augmented) | gain | p_Holm |
|---|---|---|---|---|---|---|---|
| dist_lognormal | 61.80 | Whittle (55.41) | +11.5% [+9.8,+13.2] | 2.2e-08 | Whittle-F (60.80) | +1.6% [+0.4,+2.9] | 0.031 |
| dist_negbin | 62.11 | Whittle (55.65) | +11.6% [+10.1,+13.1] | 2.2e-08 | Whittle-F (60.57) | +2.6% [+1.2,+3.8] | 0.001 |
| dist_uniformpat | 58.19 | Whittle (52.77) | +10.3% [+8.4,+12.1] | 2.2e-08 | Whittle-F (57.44) | +1.3% [-0.4,+3.2] | 0.164 |
| load_0.12 | 81.09 | Whittle (71.51) | +13.4% [+12.6,+14.3] | 2.2e-08 | Whittle-F (78.91) | +2.8% [+1.9,+3.7] | 6.9e-07 |
| load_0.14 | 72.32 | GA (64.02) | +13.0% [+11.4,+14.6] | 2.2e-08 | Whittle-F (71.12) | +1.7% [+0.8,+2.6] | 0.002 |
| load_0.18 | 51.50 | Whittle (46.82) | +10.0% [+7.7,+12.4] | 1.8e-07 | Whittle-F (50.63) | +1.7% [+0.3,+3.0] | 0.031 |
| load_0.2 | 43.70 | Whittle (39.88) | +9.6% [+7.5,+11.5] | 3.1e-07 | Whittle-F (43.51) | +0.4% [-1.5,+2.2] | 0.371 |
| main | 63.38 | Whittle (56.43) | +12.3% [+10.8,+13.8] | 2.2e-08 | Whittle-F (62.05) | +2.1% [+1.0,+3.3] | 0.002 |
| mismatch_festival | 56.99 | GA (53.02) | +7.5% [+5.3,+9.4] | 3.2e-05 | Whittle-F (55.76) | +2.2% [+0.4,+3.9] | 0.009 |
| mismatch_weekend | 60.61 | PSO (57.85) | +4.8% [+3.3,+6.3] | 4.7e-06 | Whittle-F (58.29) | +4.0% [+2.4,+5.6] | 1.2e-05 |
| nodet_mismatch_festival | 53.04 | - | - | - | Whittle-F (48.19) | +10.1% [+7.9,+12.4] | 7.5e-09 |
| nodet_mismatch_weekend | 45.08 | - | - | - | Whittle-F (32.93) | +36.9% [+32.9,+41.2] | 7.5e-09 |
| noise_0.1 | 63.35 | Whittle (56.43) | +12.3% [+10.8,+13.8] | 2.2e-08 | Whittle-F (61.90) | +2.3% [+1.3,+3.4] | 3.4e-04 |
| noise_0.2 | 63.31 | Whittle (56.43) | +12.2% [+10.6,+13.7] | 2.2e-08 | Whittle-F (61.59) | +2.8% [+1.7,+3.9] | 3.9e-05 |
| noise_0.3 | 63.07 | Whittle (56.43) | +11.8% [+10.2,+13.3] | 2.2e-08 | Whittle-F (60.80) | +3.7% [+2.4,+5.0] | 1.1e-05 |
| noise_0.5 | 60.98 | Whittle (56.43) | +8.1% [+6.4,+9.6] | 1.4e-07 | Whittle-F (58.43) | +4.4% [+2.9,+5.9] | 8.8e-06 |
| pat_20 | 66.35 | GA (58.39) | +13.6% [+12.3,+14.9] | 2.2e-08 | Whittle-F (65.03) | +2.0% [+1.3,+2.7] | 1.1e-05 |
| pat_30 | 65.24 | Whittle (58.13) | +12.2% [+10.7,+13.8] | 2.2e-08 | Whittle-F (64.22) | +1.6% [+0.6,+2.6] | 0.005 |
| pat_60 | 61.82 | Whittle (55.02) | +12.3% [+10.7,+14.0] | 2.2e-08 | Whittle-F (60.34) | +2.5% [+0.7,+4.1] | 0.003 |
| profile_festival | 59.75 | GA (53.02) | +12.7% [+10.6,+14.8] | 2.2e-08 | Whittle-F (58.25) | +2.6% [+1.4,+3.9] | 5.1e-04 |
| profile_weekend | 64.24 | PSO (57.85) | +11.1% [+9.3,+12.8] | 2.2e-08 | Whittle-F (62.89) | +2.1% [+0.9,+3.3] | 0.004 |
| scale_10 | 58.73 | Whittle (51.87) | +13.2% [+11.3,+14.7] | 0.020 | Whittle-F (56.21) | +4.5% [+3.6,+5.5] | 0.020 |
| scale_20 | 57.73 | Whittle (51.02) | +13.1% [+11.0,+15.3] | 0.020 | Whittle-F (55.53) | +4.0% [+2.2,+5.9] | 0.020 |
| scale_40 | 59.49 | Whittle (52.98) | +12.3% [+11.1,+13.4] | 0.020 | Whittle-F (57.62) | +3.3% [+2.4,+4.2] | 0.020 |
| tau_10 | 58.48 | Whittle (46.06) | +27.0% [+24.2,+29.9] | 2.2e-08 | Lag-SIPP (57.71) | +1.3% [+0.0,+2.6] | 0.050 |
| tau_2 | 66.81 | Whittle (63.07) | +5.9% [+4.8,+7.1] | 2.2e-08 | Whittle-F (64.63) | +3.4% [+2.4,+4.4] | 3.8e-06 |
| tau_8 | 60.15 | Whittle (50.85) | +18.3% [+16.3,+20.3] | 2.2e-08 | Whittle-F (59.32) | +1.4% [+0.4,+2.4] | 0.015 |

## Main benchmark: ablations (full minus variant, QASS points)

- SDQ-core: +0.05 (+0.08%), p=0.685
- abl-no-lead: +0.63 (+1.01%), p=0.035
- abl-steady: +1.66 (+2.68%), p=2.1e-04
- abl-surrogate: +3.21 (+5.35%), p=1.9e-08
- abl-no-forecast: +6.09 (+10.65%), p=1.9e-09
