# Data source

| | |
|---|---|
| File | `data/raw/ai4i2020.csv` |
| Dataset | AI4I 2020 Predictive Maintenance Dataset |
| Repository | UCI Machine Learning Repository, https://archive.ics.uci.edu/dataset/601 |
| DOI | `10.24432/C5HS5C` |
| Licence | Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/ |
| Source paper | S. Matzka, *Explainable Artificial Intelligence for Predictive Maintenance Applications*, International Conference on Artificial Intelligence for Industries (AI4I), 2020 |
| Downloaded | 4 November 2026, archive `ai4i_2020_predictive_maintenance_dataset.zip` from the UCI page (file date inside the archive: 22 May 2023) |
| Changes | none - the file is kept in its original form |
| SHA-256 | `dc6630cd9b1f0f853922fad78a1b6436570d3f1ec863f1dd5c4340ac56bc8a8e` |
| Size and format | 522 048 bytes, UTF-8 with BOM, CRLF line endings |
| Dimensions | 10 000 rows, 14 columns |

The data are synthetic and contain no personal data.

## Column dictionary

| Column | Name in code | Type | Unit | Description | Role |
|---|---|---|---|---|---|
| UDI | `udi` | int | - | Row identifier (1-10000) | identifier, not a feature |
| Product ID | `product_id` | str | - | Variant letter followed by a serial number | identifier, not a feature |
| Type | `type` | category | - | Product quality variant L / M / H | feature |
| Air temperature [K] | `air_temp_k` | float | K | Ambient air temperature | feature |
| Process temperature [K] | `process_temp_k` | float | K | Process temperature | feature |
| Rotational speed [rpm] | `rotational_speed_rpm` | int | rpm | Spindle rotational speed | feature |
| Torque [Nm] | `torque_nm` | float | Nm | Torque | feature |
| Tool wear [min] | `tool_wear_min` | int | min | Accumulated tool usage time | feature |
| Machine failure | `machine_failure` | 0/1 | - | 1 if the run ended with a machine failure | label |
| TWF, HDF, PWF, OSF, RNF | `twf` ... `rnf` | 0/1 | - | Failure mode flags (tool wear, heat dissipation, power, overstrain, random) | excluded - they define the label; used only in error analysis |

## Citation

AI4I 2020 Predictive Maintenance Dataset [Dataset]. (2020). UCI Machine Learning Repository. https://doi.org/10.24432/C5HS5C
