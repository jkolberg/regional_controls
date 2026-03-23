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


def _normalize_industry_text(value):
	text = str(value).strip().lower()
	text = text.replace("n.e.c.", "")
	text = text.replace("not elsewhere classified", "")
	text = re.sub(r"[^a-z0-9]+", " ", text)
	return " ".join(text.split())


def _extract_naics_code(value):
	if pd.isna(value):
		return pd.NA

	match = re.search(r"\b([0-9][0-9A-Z]{1,6}|[bB]{7})\b", str(value))
	if not match:
		return pd.NA

	return match.group(1).upper()


def _extract_naics_2digit(value):
	if pd.isna(value):
		return pd.NA

	text = str(value).strip().upper()
	match = re.match(r"^(\d{2})", text)
	if match:
		return int(match.group(1))

	if text.startswith("3MS"):
		return 33
	if text.startswith("4MS"):
		return 44

	return pd.NA


def _remove_leading_industry_code(value):
	if pd.isna(value):
		return value

	text = str(value)
	return re.sub(r"^\s*(?:[0-9][0-9A-Z]{1,6}|[bB]{7})\s*[-.:]?\s*", "", text)


def _pick_first_existing_column(df, candidates):
	for col in candidates:
		if col in df.columns:
			return col
	return None


def _build_industry_lookup(util):
	data_dir = Path(util.get_data_dir())
	configured_filename = _get_input_filename(util, "industry_crosswalk")
	if not configured_filename:
		raise FileNotFoundError(
			"No industry crosswalk configured. Add input_table_list tablename=industry_crosswalk in configs_pypyr/settings.yaml."
		)

	crosswalk_path = data_dir / configured_filename
	if not crosswalk_path.exists():
		raise FileNotFoundError(
			f"Configured industry crosswalk not found: {crosswalk_path}. Check configs_pypyr/settings.yaml input_table_list."
		)

	industry_crosswalk = pd.read_csv(crosswalk_path)
	remi_col = _pick_first_existing_column(industry_crosswalk, ["remi_industry", "industry_group_2nd_table"])
	naics_col = _pick_first_existing_column(industry_crosswalk, ["naics", "naics_2digit_codes"])
	industry_col = _pick_first_existing_column(industry_crosswalk, ["industry", "industry_code"])
	if not remi_col or not naics_col or not industry_col:
		raise KeyError(
			"industry_crosswalk must include REMI label, NAICS 2-digit list, and grouped industry code columns. "
			"Supported headers are remi_industry/industry_group_2nd_table, naics/naics_2digit_codes, and industry/industry_code."
		)

	def _parse_naics_list(value):
		return tuple(int(code.strip()) for code in str(value).split(",") if code.strip())

	industry_crosswalk["_naics_2digit_codes"] = industry_crosswalk[naics_col].apply(_parse_naics_list)
	industry_crosswalk["_industry_code"] = industry_crosswalk[industry_col].astype(str).str.strip().str.upper()

	industry_lookup = (
		industry_crosswalk.assign(_key=industry_crosswalk[remi_col].apply(_normalize_industry_text))
		.set_index("_key")["_industry_code"]
		.to_dict()
	)

	industry_code_xwalk = {}
	for _, row in industry_crosswalk[["_naics_2digit_codes", "_industry_code"]].iterrows():
		industry_code = row["_industry_code"]
		for two_digit_code in row["_naics_2digit_codes"]:
			industry_code_xwalk[int(two_digit_code)] = industry_code

	return industry_lookup, industry_code_xwalk


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

	pums_person_nongq["is_worker"] = pums_person_nongq["ESR"].isin([1, 2, 3, 4, 5]).astype(int)
	labor_force_participation_numerator = pums_person_nongq.loc[pums_person_nongq["is_worker"] == 1].groupby(["county_id", "age_group"])["PWGTP"].sum()
	labor_force_participation_denominator = person_weight
	labor_force_participation_rates = (labor_force_participation_numerator / labor_force_participation_denominator).fillna(0)


	return gq_rates, headship_rates, labor_force_participation_rates

def aggregate_age_groups(df):
	df = df.reset_index()
	age_group_mapping = {
		"ages_0_4": "ages_0_24",
		"ages_5_9": "ages_0_24",
		"ages_10_14": "ages_0_24",
		"ages_15_19": "ages_0_24",
		"ages_20_24": "ages_0_24",
		"ages_25_29": "ages_25_44",
		"ages_30_34": "ages_25_44",
		"ages_35_39": "ages_25_44",
		"ages_40_44": "ages_25_44",
		"ages_45_49": "ages_45_64",
		"ages_50_54": "ages_45_64",
		"ages_55_59": "ages_45_64",
		"ages_60_64": "ages_45_64",
		"ages_65_69": "ages_65_plus",
		"ages_70_74": "ages_65_plus",
		"ages_75_79": "ages_65_plus",
		"ages_80_84": "ages_65_plus",
		"ages_85_plus": "ages_65_plus",
	}

	df["age_group"] = df["age_group"].map(age_group_mapping).fillna(df["age_group"])
	return df.groupby(["county_id", "age_group"]).sum()

def build_remi_controls(util):
	pums_person = util.get_table("pums_person_prepared")
	pums_hh = util.get_table("pums_households_prepared")
	remi = util.get_table("regional_controls")
	gq_rates, headship_rates, labor_force_participation_rates = _calculate_pums_rates(pums_person, pums_hh)

	forecast_year = util.get_setting("forecast_year")
	category_col = "category" if "category" in remi.columns else "Category"
	age_col = category_col
	year_col = forecast_year

	remi_age = remi.loc[
		remi[age_col].astype(str).str.contains("ages_", na=False),
		["county_id", age_col, year_col],
	].copy()
	remi_age[year_col] = remi_age[year_col] * 1000
	remi_age = remi_age.rename(columns={age_col: "age_group", year_col: "total_pop"})
	remi_age = remi_age.set_index(["county_id", "age_group"])

	remi_age["gq"] = remi_age.index.map(gq_rates).fillna(0) * remi_age["total_pop"]
	remi_age["hhpop"] = remi_age["total_pop"] - remi_age["gq"]
	remi_age["hh"] = remi_age["hhpop"] * remi_age.index.map(headship_rates).fillna(0)
	remi_age["labor_force"] = remi_age.index.map(labor_force_participation_rates).fillna(0) * remi_age["hhpop"]
	occupation_crosswalk, occupation_code_xwalk = _build_occupation_crosswalk(util)
	category_lookup = (
		occupation_crosswalk.assign(_key=occupation_crosswalk["occupation_group_2nd_table"].apply(_normalize_occ_text))
		.set_index("_key")["occupation_code"]
		.to_dict()
	)

	category_series = remi[category_col].astype(str)
	naics_source_col = "NAICSP_2digit" if "NAICSP_2digit" in pums_person.columns else "NAICSP"
	if naics_source_col == "NAICSP_2digit":
		pums_person["naics_2digit"] = pd.to_numeric(pums_person[naics_source_col], errors="coerce").astype("Int64")
	else:
		pums_person["naics_2digit"] = pums_person[naics_source_col].apply(_extract_naics_2digit).astype("Int64")
	occupation_text = category_series.str.extract(r"Employment by Occupation\s*-\s*(.*)$", expand=False)
	occupation_key = occupation_text.apply(lambda x: _normalize_occ_text(x) if pd.notna(x) else x)
	direct_code = occupation_key.map(category_lookup)
	soc_two_digit = pd.to_numeric(
		occupation_text.str.extract(r"(\d{2})(?:-0000)?")[0],
		errors="coerce",
	)
	fallback_code = soc_two_digit.map(occupation_code_xwalk)
	remi["occupation_code"] = direct_code.fillna(fallback_code).astype("Int64")
	industry_lookup, industry_code_xwalk = _build_industry_lookup(util)
	pums_person["industry_code"] = pums_person["naics_2digit"].map(industry_code_xwalk)
	industry_text = category_series.str.extract(r"Employment(?:\s+by\s+Major\s+Industry)?\s*-\s*(.*)$", expand=False)
	industry_label = industry_text.apply(_remove_leading_industry_code)
	industry_key = industry_label.apply(lambda x: _normalize_industry_text(x) if pd.notna(x) else x)
	direct_industry_code = industry_key.map(industry_lookup)
	remi_naics_2digit = pd.to_numeric(industry_text.str.extract(r"(\d{2})")[0], errors="coerce")
	fallback_industry_code = remi_naics_2digit.map(industry_code_xwalk)
	existing_industry_code = category_series.str.extract(r"^naics_(.+)$", expand=False)
	remi["industry_code"] = existing_industry_code.fillna(direct_industry_code).fillna(fallback_industry_code)
	major_industry_mask = remi[category_col].astype(str).str.contains(
		r"Employment(?:\s+by\s+Major\s+Industry)?\s*-|^naics_",
		na=False,
		regex=True,
	)
	if major_industry_mask.any() and remi.loc[major_industry_mask, "industry_code"].notna().sum() == 0:
		raise ValueError(
			"Could not map REMI industry rows to industry_code. Update data/industry_crosswalk.csv remi_industry labels (or industry_group_2nd_table) to match REMI rows and confirm naics mappings are provided."
		)

	remi_emp = remi.loc[
		remi[category_col].str.contains("Employment by Occupation", na=False),
		["county_id", "occupation_code", year_col],
	].copy()
	remi_emp = remi_emp.loc[remi_emp["occupation_code"].notna()].copy()
	remi_emp = remi_emp.rename(columns={year_col: "employment"})
	remi_emp["employment"] = remi_emp["employment"] * 1000
	remi_ind = remi.loc[
		major_industry_mask,
		["county_id", "industry_code", year_col],
	].copy()
	remi_ind = remi_ind.loc[remi_ind["industry_code"].notna()].copy()
	remi_ind = remi_ind.rename(columns={year_col: "employment"})
	remi_ind["employment"] = remi_ind["employment"] * 1000

	labor_force = remi_age.groupby("county_id")["labor_force"].sum()

	remi_emp["labor_force"] = (
		remi_emp["employment"]
		/ remi_emp.groupby("county_id")["employment"].transform("sum")
		* remi_emp["county_id"].map(labor_force)
	)
	remi_ind["labor_force"] = (
		remi_ind["employment"]
		/ remi_ind.groupby("county_id")["employment"].transform("sum")
		* remi_ind["county_id"].map(labor_force)
	)

	# Multiple REMI rows can collapse into the same grouped bucket (e.g., industry 92),
	# so aggregate to unique county/control keys before pivoting.
	remi_emp = (
		remi_emp.groupby(["county_id", "occupation_code"], as_index=False)["labor_force"]
		.sum()
	)
	remi_ind = (
		remi_ind.groupby(["county_id", "industry_code"], as_index=False)["labor_force"]
		.sum()
	)

	remi_hh = remi_age.reset_index().groupby("county_id")["hh"].sum()
	remi_age = aggregate_age_groups(remi_age).copy()
	out = remi_age["hhpop"].unstack()
	out["num_hh"] = out.index.map(remi_hh)

	remi_emp["occupation_code"] = "soc_" + remi_emp["occupation_code"].astype(str)
	emp_out = remi_emp.set_index(["county_id", "occupation_code"])["labor_force"].unstack()
	out = out.merge(emp_out, left_index=True, right_index=True, how="left")
	remi_ind["industry_code"] = "naics_" + remi_ind["industry_code"].astype(str)
	industry_out = remi_ind.set_index(["county_id", "industry_code"])["labor_force"].unstack()
	out = out.merge(industry_out, left_index=True, right_index=True, how="left")
	out = out.fillna(0).round(0).astype(int)

	out.index.name = "county_id"
	util.save_table("county_controls", out.reset_index())


def run_step(context):
	print("Generating county controls from REMI and prepared PUMS...")
	util = Util(settings_path=context["configs_dir"])
	build_remi_controls(util)
	return context
