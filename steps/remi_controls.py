from pathlib import Path
import re

import pandas as pd

from util import Util


def _normalize_remi_age_category(value):
	if pd.isna(value):
		return value

	text = str(value)
	existing_label = re.search(r"ages_(?:\d+_\d+|85_plus)", text)
	if existing_label:
		return existing_label.group(0)

	range_match = re.search(r"(\d+)\s*(?:-|to)\s*(\d+)", text, flags=re.IGNORECASE)
	plus_match = re.search(r"(\d+)\s*\+", text)

	if range_match:
		start_age = int(range_match.group(1))
	elif plus_match:
		start_age = int(plus_match.group(1))
	else:
		return text

	if start_age >= 85:
		return "ages_85_plus"
	if start_age == 0:
		return "ages_0_4"
	return f"ages_{start_age}_{start_age + 5}"


def _build_occupation_crosswalk():
	occupation_crosswalk = pd.DataFrame(
		[
			("Management, business, and financial operations occupations", "11-0000,13-0000"),
			("Computer, mathematical, architecture, and engineering occupations", "15-0000,17-0000"),
			("Life, physical, and social science occupations", "19-0000"),
			("Community and social service occupations", "21-0000"),
			("Legal occupations", "23-0000"),
			("Educational instruction and library occupations", "25-0000"),
			("Arts, design, entertainment, sports, and media occupations", "27-0000"),
			("Healthcare occupations", "29-0000,31-0000"),
			("Protective service occupations", "33-0000"),
			("Food preparation and serving related occupations", "35-0000"),
			("Building and grounds cleaning and maintenance, personal care and service occupations", "37-0000,39-0000"),
			("Sales and related, office and administrative support occupations", "41-0000,43-0000"),
			("Farming, fishing, and forestry occupations", "45-0000"),
			("Construction and extraction occupations", "47-0000"),
			("Installation, maintenance, and repair occupations", "49-0000"),
			("Production occupations", "51-0000"),
			("Transportation and material moving occupations", "53-0000"),
			("Military", "55-0000"),
		],
		columns=["occupation_group_2nd_table", "soc_codes"],
	)
	occupation_crosswalk["occupation_code"] = (
		occupation_crosswalk["soc_codes"]
		.str.replace("-0000", "", regex=False)
		.str.replace(",", "", regex=False)
	)

	occupation_code_xwalk = {}
	for _, row in occupation_crosswalk[["soc_codes", "occupation_code"]].iterrows():
		grouped_code = int(row["occupation_code"])
		for soc_code in row["soc_codes"].split(","):
			two_digit_code = int(soc_code.split("-")[0])
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


def _load_remi_table(util):
	data_dir = Path(util.get_data_dir())
	configured_filename = _get_input_filename(util, "regional_controls")
	if configured_filename:
		remi_path = data_dir / configured_filename
	else:
		remi_path = next(iter(sorted(data_dir.glob("REMI*.xlsx"))), None)

	if remi_path is None:
		raise FileNotFoundError(
			f"No REMI workbook found in {data_dir}. Configure input_table_list tablename=regional_controls."
		)
	if not remi_path.exists():
		raise FileNotFoundError(
			f"Configured REMI workbook not found: {remi_path}. Check configs_pypyr/settings.yaml input_table_list."
		)

	remi = pd.read_excel(remi_path, skiprows=5)
	county_map = {
		"King County": 53033,
		"Kitsap County": 53035,
		"Pierce County": 53053,
		"Snohomish County": 53061,
	}
	remi["county_id"] = remi["Region"].map(county_map)
	remi = remi.loc[remi["county_id"].notna()].copy()
	remi["county_id"] = remi["county_id"].astype(int)
	remi["Category"] = remi["Category"].apply(_normalize_remi_age_category)
	return remi


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
	remi = _load_remi_table(util)
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

	occupation_crosswalk, occupation_code_xwalk = _build_occupation_crosswalk()
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
