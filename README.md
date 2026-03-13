# regional-controls

Pipeline to build county-level PopulationSim controls from PUMS + REMI, run PopulationSim, archive core outputs by forecast scenario, and generate summary charts.

## What This Project Does

This project runs a pypyr workflow that:

1. Builds/loads geography and PUMS inputs.
2. Loads configured source tables into `data/pipeline.h5`.
3. Loads and normalizes a configured REMI workbook from `regional_forecast` in settings.
4. Computes Popsim controls by county and writes `data/remi_controls.csv`.
5. Runs PopulationSim.
6. Archives key output files into `output/archive/<regional_forecast_name>/`.
7. Converts an UrbanSim cache to HDF for comparison controls.
8. Writes summary plots to `output/results_summaries/`.

## Project Layout

- `run.py`: pypyr entrypoint.
- `configs_pypyr/settings.yaml`: primary pipeline settings and step order.
- `configs_popsim/`: PopulationSim configuration.
- `steps/`: pypyr step modules.
- `utils/util.py`: shared utility class for settings + HDF I/O.
- `data/`: inputs + pipeline store + controls output.
- `output/`: PopulationSim outputs, archives, summary charts.

## Requirements

- Python 3.11+
- Windows path examples are used in this repo.
- Dependencies are managed in `pyproject.toml`.

Core dependencies include:

- `pypyr`
- `pandas`
- `openpyxl`
- `populationsim`
- `matplotlib`

## Setup

If using uv:

```bash
uv sync
```

## Configuration

Edit `configs_pypyr/settings.yaml`.

Important keys:

- `regional_forecasts_dir`: network/local directory containing REMI workbooks.
- `regional_forecast`:
	- `tablename: regional_controls`
	- `filename: <scenario workbook>.xlsx`
- `county_map`: maps REMI region names to county IDs.
- `pums_table_list`: PUMS and PUMA geography CSVs.
- `input_table_list`: additional CSV inputs (for example occupation crosswalk).
- `output_table_list`: CSVs written from HDF pipeline tables.
- `urbansim_baseyear_cache`: base-year cache folder for cache conversion step.
- `comparison_controls_store`: HDF path used for chart comparison controls.
- `comparison_controls_table`: HDF table name (default `/annual_household_control_totals`).
- `steps`: pypyr step order.

## Run

Run the full configured pipeline:

```
uv run run.py
```
