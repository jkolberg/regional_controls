from pathlib import Path
import re

import pandas as pd

from utils import Util


def _build_occupation_crosswalk(util):
	data_dir = Path(util.get_data_dir())
	configured_filename = _get_input_filename(util, "occupation_crosswalk")
	if not configured_filename:
		raise FileNotFoundError(
			"No occupation crosswalk configured. Add input_table_list tablename=occupation_crosswalk in configs_pypyr/settings.yaml."
		)

	crosswalk_path = data_dir / configured_filename
	if not crosswalk_path.exists():
		raise FileNotFoundError(
			f"Configured occupation crosswalk not found: {crosswalk_path}. Check configs_pypyr/settings.yaml input_table_list."
		)

	occupation_crosswalk = pd.read_csv(crosswalk_path)
	occupation_crosswalk["soc_2digit_codes"] = occupation_crosswalk["soc_2digit_codes"].apply(
		lambda value: tuple(int(code.strip()) for code in str(value).split(",") if code.strip())
	)

	occupation_code_xwalk = {}
	for _, row in occupation_crosswalk[["soc_2digit_codes", "occupation_code"]].iterrows():
		grouped_code = int(row["occupation_code"])
		for two_digit_code in row["soc_2digit_codes"]:
			occupation_code_xwalk[two_digit_code] = grouped_code

	return occupation_crosswalk, occupation_code_xwalk


def _normalize_occ_text(value):
	text = str(value).strip().lower()
	text = text.replace("deaning", "cleaning")
	text = re.sub(r"[^a-z0-9]+", " ", text)
	return " ".join(text.split())


def _get_input_filename(util, tablename):
	for table in util.get_table_list():
		if table.get("tablename") == tablename:
			return table.get("filename")
	return None


def _calculate_pums_rates(pums_person, pums_hh):
	gq_numerator = pums_person.loc[pums_person["gq"] == 1].groupby(["county_id", "age_group"])["PWGTP"].sum()
	gq_denominator = pums_person.groupby(["county_id", "age_group"])["PWGTP"].sum()
	gq_rates = (gq_numerator / gq_denominator).fillna(0)

	if "TYPEHUGQ" in pums_hh.columns:
		pums_hh_nongq = pums_hh.loc[pums_hh["TYPEHUGQ"] == 1].copy()
	else:
		pums_hh_nongq = pums_hh.loc[pums_hh["TYPE"].isin([1, 2])].copy()
	pums_person_nongq = pums_person.loc[pums_person["gq"] == 0].copy()

	hh_weight = pums_hh_nongq.groupby(["county_id", "age_head_group"])["WGTP"].sum()
	person_weight = pums_person_nongq.groupby(["county_id", "age_group"])["PWGTP"].sum()
	hh_weight.index = hh_weight.index.set_names(["county_id", "age_group"])
	headship_rates = (hh_weight / person_weight).fillna(0)

	return gq_rates, headship_rates


def build_remi_controls(util):
	pums_person = util.get_table("pums_person_prepared")
	pums_hh = util.get_table("pums_households_prepared")
	remi = util.get_table("regional_controls")
	gq_rates, headship_rates = _calculate_pums_rates(pums_person, pums_hh)

	age_col = "Category"
	year_col = 2050

	remi_age = remi.loc[
		remi[age_col].astype(str).str.contains("ages_", na=False),
		["county_id", age_col, year_col],
	].copy()
	remi_age[year_col] = remi_age[year_col] * 1000
	remi_age = remi_age.rename(columns={age_col: "age_group", year_col: "total_pop_2050"})
	remi_age = remi_age.set_index(["county_id", "age_group"])

	remi_age["gq_2050"] = remi_age.index.map(gq_rates).fillna(0) * remi_age["total_pop_2050"]
	remi_age["hhpop_2050"] = remi_age["total_pop_2050"] - remi_age["gq_2050"]
	remi_age["hh_2050"] = remi_age["hhpop_2050"] * remi_age.index.map(headship_rates).fillna(0)

	occupation_crosswalk, occupation_code_xwalk = _build_occupation_crosswalk(util)
	category_col = "category" if "category" in remi.columns else "Category"
	category_lookup = (
		occupation_crosswalk.assign(_key=occupation_crosswalk["occupation_group_2nd_table"].apply(_normalize_occ_text))
		.set_index("_key")["occupation_code"]
		.to_dict()
	)

	category_series = remi[category_col].astype(str)
	occupation_text = category_series.str.extract(r"Employment by Occupation\s*-\s*(.*)$", expand=False)
	occupation_key = occupation_text.apply(lambda x: _normalize_occ_text(x) if pd.notna(x) else x)
	direct_code = occupation_key.map(category_lookup)
	soc_two_digit = pd.to_numeric(
		occupation_text.str.extract(r"(\d{2})(?:-0000)?")[0],
		errors="coerce",
	)
	fallback_code = soc_two_digit.map(occupation_code_xwalk)
	remi["occupation_code"] = direct_code.fillna(fallback_code).astype("Int64")

	remi_emp = remi.loc[
		remi["Category"].str.contains("Employment by Occupation", na=False),
		["county_id", "occupation_code", year_col],
	].copy()
	remi_emp = remi_emp.loc[remi_emp["occupation_code"].notna()].copy()
	remi_emp = remi_emp.rename(columns={year_col: "employment_2050"})
	remi_emp["employment_2050"] = remi_emp["employment_2050"] * 1000

	labor_force = remi.loc[
		remi["Category"].str.contains("Labor Force", na=False),
		["county_id", year_col],
	].drop_duplicates(subset=["county_id"])
	labor_force = (labor_force.set_index("county_id")[year_col] * 1000)

	remi_emp["labor_force_2050"] = (
		remi_emp["employment_2050"]
		/ remi_emp.groupby("county_id")["employment_2050"].transform("sum")
		* remi_emp["county_id"].map(labor_force)
	)

	remi_hh = remi_age.reset_index().groupby("county_id")["hh_2050"].sum()
	out = remi_age["hhpop_2050"].unstack()
	out["num_hh"] = out.index.map(remi_hh)

	remi_emp["occupation_code"] = "soc_" + remi_emp["occupation_code"].astype(str)
	emp_out = remi_emp.set_index(["county_id", "occupation_code"])["labor_force_2050"].unstack()
	out = out.merge(emp_out, left_index=True, right_index=True, how="left")
	out = out.fillna(0).round(0).astype(int)

	out.index.name = "county_id"
	util.save_table("county_controls", out.reset_index())


def run_step(context):
	print("Generating county controls from REMI and prepared PUMS...")
	util = Util(settings_path=context["configs_dir"])
	build_remi_controls(util)
	return context
