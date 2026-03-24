from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import Util


def _get_repo_root() -> Path:
	return Path(__file__).resolve().parent.parent


def _get_input_filename(util: Util, table_name: str) -> str:
	for table in util.settings.get("input_table_list", []):
		if table["tablename"] == table_name:
			return table["filename"]
	raise KeyError(f"Input table '{table_name}' not found in settings.yaml")


def _get_regional_controls_stem(util: Util) -> str:
	for table in util.settings.get("regional_forecast", []):
		if table.get("tablename") == "regional_controls":
			return Path(table["filename"]).stem

	return Path(_get_input_filename(util, "regional_controls")).stem


def _build_income_bins(pums_hh: pd.DataFrame):
	weighted_income = pums_hh[["HINCP", "WGTP"]].dropna().copy()
	weighted_income = weighted_income.loc[weighted_income["WGTP"] > 0].sort_values("HINCP")
	cum_weight_share = weighted_income["WGTP"].cumsum() / weighted_income["WGTP"].sum()
	weighted_cutoffs = np.interp([0.25, 0.5, 0.75], cum_weight_share, weighted_income["HINCP"])

	income_bin_edges = [-np.inf, *weighted_cutoffs.tolist(), np.inf]
	cutoff_k = np.rint(weighted_cutoffs / 1000).astype(int)
	income_bin_labels = [
		f"up to ${cutoff_k[0]:,}k",
		f"${cutoff_k[0]:,}k to ${cutoff_k[1]:,}k",
		f"${cutoff_k[1]:,}k to ${cutoff_k[2]:,}k",
		f"${cutoff_k[2]:,}k+",
	]
	return income_bin_edges, income_bin_labels


def _load_control_totals(util: Util) -> pd.DataFrame:
	store_path = util.settings.get("comparison_controls_store")
	table_name = util.settings.get("comparison_controls_table", "/annual_household_control_totals")
	if not store_path:
		raise KeyError("comparison_controls_store is not configured in settings.yaml")

	store_path = _get_repo_root() / store_path
	with pd.HDFStore(str(store_path)) as store:
		return store[table_name]


def build_results_summary(util: Util):
	repo_root = _get_repo_root()
	synthetic_households_path = repo_root / "output" / "synthetic_households.csv"
	if not synthetic_households_path.exists():
		raise FileNotFoundError(
			f"Synthetic households file not found: {synthetic_households_path}. Run PopulationSim first."
		)

	forecast_year = util.get_setting("forecast_year")
	pums_hh = util.get_table("seed_households").copy()
	synthetic_hh = pd.read_csv(synthetic_households_path)
	control_totals = _load_control_totals(util)

	income_bin_edges, income_bin_labels = _build_income_bins(pums_hh)
	pums_hh["income_quartile"] = pd.cut(
		pums_hh["HINCP"],
		bins=income_bin_edges,
		labels=[1, 2, 3, 4],
		include_lowest=True,
	).astype("Int64")
	synthetic_hh["income"] = pd.cut(
		synthetic_hh["HINCP"],
		bins=income_bin_edges,
		labels=income_bin_labels,
		include_lowest=True,
	)

	pums_hh["hhsz"] = np.where(pums_hh["NP"] > 7, 7, pums_hh["NP"])
	synthetic_hh["hhsz"] = np.where(synthetic_hh["NP"] > 7, 7, synthetic_hh["NP"])
	size_categories = list(range(1, 8))
	size_labels = [str(i) for i in range(1, 7)] + ["7+"]

	pums_hh["workers"] = np.where(pums_hh["workers"] > 4, 4, pums_hh["workers"])
	synthetic_hh["workers"] = np.where(synthetic_hh["workers"] > 4, 4, synthetic_hh["workers"])
	worker_categories = [0, 1, 2, 3, 4]
	worker_labels = ["0", "1", "2", "3", "4+"]

	pums_income_counts = (
		pums_hh.groupby("income_quartile", observed=False)["WGTP"]
		.sum()
		.reindex([1, 2, 3, 4], fill_value=0)
		.values
	)
	synthetic_income_counts = (
		synthetic_hh["income"].value_counts().reindex(income_bin_labels, fill_value=0).values
	)
	pums_size_counts = (
		pums_hh.groupby("hhsz")["WGTP"].sum().reindex(size_categories, fill_value=0).values
	)
	synthetic_size_counts = (
		synthetic_hh.groupby("hhsz").size().reindex(size_categories, fill_value=0).values
	)
	pums_worker_counts = (
		pums_hh.groupby("workers")["WGTP"].sum().reindex(worker_categories, fill_value=0).values
	)
	synthetic_worker_counts = (
		synthetic_hh.groupby("workers").size().reindex(worker_categories, fill_value=0).values
	)

	income_compare = pd.DataFrame(
		{
			"2024": control_totals.loc[control_totals.index == 2024]
			.groupby("income_min")["total_number_of_households"]
			.sum(),
			str(forecast_year): control_totals.loc[control_totals.index == forecast_year]
			.groupby("income_min")["total_number_of_households"]
			.sum(),
		}
	).fillna(0).sort_index()
	persons_compare = pd.DataFrame(
		{
			"2024": control_totals.loc[control_totals.index == 2024]
			.groupby("persons_min")["total_number_of_households"]
			.sum(),
			str(forecast_year): control_totals.loc[control_totals.index == forecast_year]
			.groupby("persons_min")["total_number_of_households"]
			.sum(),
		}
	).fillna(0).sort_index()
	workers_compare = pd.DataFrame(
		{
			"2024": control_totals.loc[control_totals.index == 2024]
			.groupby("workers_min")["total_number_of_households"]
			.sum(),
			str(forecast_year): control_totals.loc[control_totals.index == forecast_year]
			.groupby("workers_min")["total_number_of_households"]
			.sum(),
		}
	).fillna(0).sort_index()

	fig, axes = plt.subplots(2, 3, figsize=(22, 12))
	width = 0.4

	x_income = np.arange(len(income_bin_labels))
	axes[0, 0].bar(x_income - width / 2, pums_income_counts, width=width, label="PUMS", color="steelblue")
	axes[0, 0].bar(
		x_income + width / 2,
		synthetic_income_counts,
		width=width,
		label="Synthetic {}".format(forecast_year),
		color="darkorange",
	)
	axes[0, 0].set_xticks(x_income)
	axes[0, 0].set_xticklabels(income_bin_labels, rotation=20, ha="right")
	axes[0, 0].set_xlabel("Income Group")
	axes[0, 0].set_ylabel("Number of Households")
	axes[0, 0].set_title("Income Group Comparison")
	axes[0, 0].legend()

	x_size = np.arange(len(size_categories))
	axes[0, 1].bar(x_size - width / 2, pums_size_counts, width=width, label="PUMS", color="steelblue")
	axes[0, 1].bar(
		x_size + width / 2,
		synthetic_size_counts,
		width=width,
		label="Synthetic {}".format(forecast_year),
		color="darkorange",
	)
	axes[0, 1].set_xticks(x_size)
	axes[0, 1].set_xticklabels(size_labels)
	axes[0, 1].set_xlabel("Household Size")
	axes[0, 1].set_ylabel("Number of Households")
	axes[0, 1].set_title("Household Size Comparison")
	axes[0, 1].legend()

	x_workers = np.arange(len(worker_categories))
	axes[0, 2].bar(
		x_workers - width / 2,
		pums_worker_counts,
		width=width,
		label="PUMS",
		color="steelblue",
	)
	axes[0, 2].bar(
		x_workers + width / 2,
		synthetic_worker_counts,
		width=width,
		label="Synthetic {}".format(forecast_year),
		color="darkorange",
	)
	axes[0, 2].set_xticks(x_workers)
	axes[0, 2].set_xticklabels(worker_labels)
	axes[0, 2].set_xlabel("Workers in Household")
	axes[0, 2].set_ylabel("Number of Households")
	axes[0, 2].set_title("Workers per Household Comparison")
	axes[0, 2].legend()

	income_compare.plot(kind="bar", ax=axes[1, 0], color=["steelblue", "darkorange"])
	axes[1, 0].set_xlabel("income_min")
	axes[1, 0].set_ylabel("Total Number of Households")
	axes[1, 0].set_title(f"Control Totals by Income: 2024 vs {forecast_year}")
	axes[1, 0].tick_params(axis="x", rotation=45)

	persons_compare.plot(kind="bar", ax=axes[1, 1], color=["steelblue", "darkorange"])
	axes[1, 1].set_xlabel("persons_min")
	axes[1, 1].set_ylabel("Total Number of Households")
	axes[1, 1].set_title(f"Control Totals by Household Size: 2024 vs {forecast_year}")
	axes[1, 1].tick_params(axis="x", rotation=0)

	workers_compare.plot(kind="bar", ax=axes[1, 2], color=["steelblue", "darkorange"])
	axes[1, 2].set_xlabel("workers_min")
	axes[1, 2].set_ylabel("Total Number of Households")
	axes[1, 2].set_title(f"Control Totals by Workers: 2024 vs {forecast_year}")
	axes[1, 2].tick_params(axis="x", rotation=0)

	output_dir = repo_root / "output" / "results_summaries"
	output_dir.mkdir(parents=True, exist_ok=True)
	today = pd.Timestamp.today().strftime("%Y%m%d")
	save_path = output_dir / f"{_get_regional_controls_stem(util)}_{forecast_year}_{today}.png"

	fig.suptitle("Household Comparison Summaries", y=1.02)
	fig.tight_layout()
	fig.savefig(str(save_path), dpi=300, bbox_inches="tight")
	plt.close(fig)

	if not save_path.exists() or save_path.stat().st_size == 0:
		print(f"WARNING: PNG file was not written or is empty: {save_path}")
	else:
		print(f"Saved summary chart image to: {save_path} ({save_path.stat().st_size:,} bytes)")


def run_step(context):
	print("Creating results summary charts...")
	util = Util(settings_path=context["configs_dir"])
	build_results_summary(util)
	return context
