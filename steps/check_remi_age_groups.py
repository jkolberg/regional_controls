from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from utils import Util


def _get_repo_root() -> Path:
	return Path(__file__).resolve().parent.parent


def _age_sort_key(value: str) -> int:
	match = re.search(r"_(\d+)", str(value))
	if match:
		return int(match.group(1))
	return 999


def _build_age_proportion_comparison(util: Util, base_year: int) -> pd.DataFrame:
	remi_df = util.get_table("regional_controls").copy()
	category_col = "Category" if "Category" in remi_df.columns else "category"

	remi_age = remi_df.loc[
		remi_df[category_col].astype(str).str.startswith("ages"),
		[category_col, base_year],
	].copy()
	remi_age_prop = remi_age.groupby(category_col)[base_year].sum()
	remi_age_prop = remi_age_prop / remi_age_prop.sum()

	pums_person = util.get_table("pums_person_prepared")
	pums_person = pums_person.query("gq == 0")
	pums_age_prop = pums_person.groupby("age_group").size() / len(pums_person)

	comparison = pd.DataFrame({
		f"REMI_{base_year}": remi_age_prop,
		f"PUMS_{base_year}": pums_age_prop,
	}).fillna(0)

	comparison = comparison.assign(
		sort=lambda df: df.index.to_series().apply(_age_sort_key)
	).sort_values("sort").drop(columns="sort")

	return comparison


def build_age_group_summary(util: Util) -> Path:
	base_year = int(util.get_setting("base_year"))
	comparison = _build_age_proportion_comparison(util, base_year=base_year)

	ax = comparison.plot.bar(figsize=(12, 6), color=["steelblue", "darkorange"])
	ax.set_xlabel("Age Group")
	ax.set_ylabel("Proportion of Population")
	ax.set_title(f"REMI vs PUMS Age Group Proportions ({base_year})")
	ax.tick_params(axis="x", rotation=45)
	ax.legend(title="")
	plt.tight_layout()

	output_dir = _get_repo_root() / "output" / "results_summaries"
	output_dir.mkdir(parents=True, exist_ok=True)
	today = pd.Timestamp.today().strftime("%Y%m%d")
	save_path = output_dir / f"age_dist_comparison_{today}.png"
	plt.savefig(save_path, dpi=300, bbox_inches="tight")
	plt.close()

	return save_path


def run_step(context):
	print("Checking REMI vs PUMS age group proportions...")
	util = Util(settings_path=context["configs_dir"])
	save_path = build_age_group_summary(util)
	print(f"Saved age-group comparison chart to: {save_path}")
	return context
