import re
import numpy as np
import pandas as pd

from utils import Util
from steps.prepare_pums import (
	_build_industry_lookup,
	_build_occupation_crosswalk,
	_normalize_industry_text,
)


def _normalize_occ_text(value):
	text = str(value).strip().lower()
	text = text.replace("deaning", "cleaning")
	text = re.sub(r"[^a-z0-9]+", " ", text)
	return " ".join(text.split())


def _remove_leading_industry_code(value):
	if pd.isna(value):
		return value

	text = str(value)
	return re.sub(r"^\s*(?:[0-9][0-9A-Z]{1,6}|[bB]{7})\s*[-.:]?\s*", "", text)


def _calculate_pums_rates(pums_person, pums_hh):
	gq_numerator = pums_person.loc[pums_person["gq"] == 1].groupby(["county_id", "age_group"])["PWGTP"].sum()
	gq_denominator = pums_person.groupby(["county_id", "age_group"])["PWGTP"].sum()
	gq_rates = (gq_numerator / gq_denominator).fillna(0)

	pums_hh_nongq = pums_hh.loc[pums_hh["gq"] == 0].copy()
	pums_person_nongq = pums_person.loc[pums_person["gq"] == 0].copy()

	hh_weight = pums_hh_nongq.groupby(["county_id", "age_head_group"])["WGTP"].sum()
	person_weight = pums_person_nongq.groupby(["county_id", "age_group"])["PWGTP"].sum()
	hh_weight.index = hh_weight.index.set_names(["county_id", "age_group"])
	headship_rates = (hh_weight / person_weight).fillna(0)

	pums_person_nongq["is_worker"] = pums_person_nongq["ESR"].isin([1, 2, 3, 4, 5]).astype(int)
	labor_force_participation_numerator = pums_person_nongq.loc[pums_person_nongq["is_worker"] == 1].groupby(["county_id", "age_group"])["PWGTP"].sum()
	labor_force_participation_denominator = person_weight
	labor_force_participation_rates = (labor_force_participation_numerator / labor_force_participation_denominator).fillna(0)

	hhsz_numerator = pums_hh_nongq.groupby(['county_id','age_head_group','hhsz'])['WGTP'].sum()
	hhsz_numerator.index = hhsz_numerator.index.set_names(['county_id', 'age_group', 'hhsz'])
	hhsz_denominator = hh_weight
	hhsz_rates = (hhsz_numerator / hhsz_denominator).fillna(0)

	workers_per_hhsz_numerator = pums_hh_nongq.groupby(['county_id','age_head_group','hhsz','workers'])['WGTP'].sum()
	workers_per_hhsz_denominator = pums_hh_nongq.groupby(['county_id','age_head_group','hhsz'])['WGTP'].sum()
	workers_per_hhsz_rates = workers_per_hhsz_numerator / workers_per_hhsz_denominator
	workers_per_hhsz_rates.index = workers_per_hhsz_rates.index.set_names(['county_id', 'age_group', 'hhsz', 'workers'])

	return gq_rates, headship_rates, labor_force_participation_rates, hhsz_rates, workers_per_hhsz_rates

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

def _process_occupation_codes(remi, util, category_col, year_col):
	occupation_crosswalk, occupation_code_xwalk = _build_occupation_crosswalk(util)
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
		remi[category_col].str.contains("Employment by Occupation", na=False),
		["county_id", "occupation_code", year_col],
	].copy()
	remi_emp = remi_emp.loc[remi_emp["occupation_code"].notna()].copy()
	remi_emp = remi_emp.rename(columns={year_col: "employment"})
	remi_emp["employment"] = remi_emp["employment"] * 1000

	# Multiple REMI rows can collapse into the same grouped bucket,
	# so aggregate to unique county/control keys before pivoting.
	remi_emp = (
		remi_emp.groupby(["county_id", "occupation_code"], as_index=False)["employment"]
		.sum()
	)
	return remi_emp


def _process_industry_codes(remi, util, category_col, year_col):
	category_series = remi[category_col].astype(str)
	industry_lookup, industry_code_xwalk = _build_industry_lookup(util)
	industry_text = category_series.str.extract(r"Employment(?:\s+by\s+Major\s+Industry)?\s*-\s*(.*)$", expand=False)
	industry_label = industry_text.apply(_remove_leading_industry_code)
	industry_key = industry_label.apply(lambda x: _normalize_industry_text(x) if pd.notna(x) else x)
	direct_industry_code = industry_key.map(industry_lookup)
	remi_naics_2digit = pd.to_numeric(industry_text.str.extract(r"(\d{2})")[0], errors="coerce")
	fallback_industry_code = remi_naics_2digit.map(industry_code_xwalk)
	existing_industry_code = category_series.str.extract(r"^naics_(.+)$", expand=False)
	remi["industry"] = existing_industry_code.fillna(direct_industry_code).fillna(fallback_industry_code)

	major_industry_mask = remi[category_col].astype(str).str.contains(
		r"Employment(?:\s+by\s+Major\s+Industry)?\s*-|^naics_",
		na=False,
		regex=True,
	)
	if major_industry_mask.any() and remi.loc[major_industry_mask, "industry"].notna().sum() == 0:
		raise ValueError(
			"Could not map REMI industry rows to industry codes. Update data/industry_crosswalk.csv remi_industry labels (or industry_group_2nd_table) to match REMI rows and confirm naics mappings are provided."
		)

	remi_ind = remi.loc[
		major_industry_mask,
		["county_id", "industry", year_col],
	].copy()
	remi_ind = remi_ind.loc[remi_ind["industry"].notna()].copy()
	remi_ind = remi_ind.rename(columns={year_col: "employment"})
	remi_ind["employment"] = remi_ind["employment"] * 1000

	# Multiple REMI rows can collapse into the same grouped bucket (e.g., industry 92),
	# so aggregate to unique county/control keys before pivoting.
	remi_ind = (
		remi_ind.groupby(["county_id", "industry"], as_index=False)["employment"]
		.sum()
	)
	return remi_ind

def get_non_gq_workers(pums_person,group_col):
    non_gq_workers = pums_person.query('is_worker == 1 and gq == 0').groupby(group_col)['PWGTP'].sum()
    return non_gq_workers

def get_workers_to_remi_emp_ratio(util, non_gq_workers, year_col, category_col):
    remi = util.get_table('regional_controls')
    remi_ind = _process_industry_codes(remi, util, category_col, year_col)
    remi_ind['industry'] = pd.to_numeric(remi_ind['industry'], errors='coerce')
    remi_ind = remi_ind.groupby(['industry'])['employment'].sum()
    non_gq_workers.index.names = ['industry']
    return non_gq_workers / remi_ind

def get_workers_to_remi_emp_ratio_occ(util, non_gq_workers, year_col, category_col):
    remi = util.get_table('regional_controls')
    remi_occ = _process_occupation_codes(remi, util, category_col, year_col)
    remi_occ = remi_occ.rename(columns={'occupation_code': 'occupation'}).groupby(['occupation'])['employment'].sum()
    non_gq_workers.index.names = ['occupation']
    return non_gq_workers / remi_occ

def build_remi_controls(util):
	pums_person = util.get_table("pums_person_prepared")
	pums_hh = util.get_table("pums_households_prepared")
	remi = util.get_table("regional_controls")
	gq_rates, headship_rates, labor_force_participation_rates, hhsz_rates, workers_per_hhsz_rates = _calculate_pums_rates(pums_person, pums_hh)

	forecast_year = util.get_setting("forecast_year")
	category_col = "category" if "category" in remi.columns else "Category"
	age_col = category_col
	year_col = forecast_year

	# Process age groups and calculate gq, hh, hhpop - all coming from REMI pop by age
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
	
	# sum totals for use in control totals process
	region_summed = (
		remi_age[['total_pop','hhpop','gq','hh']].sum().round(0).astype(int)
		.to_frame(name=f'{forecast_year}').reset_index(names='variable')
	)

	# Calculate labor force by industry and occupation using Pums-workers to REMI-employment ratios
	remi_emp = _process_occupation_codes(remi, util, category_col, year_col)
	remi_ind = _process_industry_codes(remi, util, category_col, year_col)
	
	# sum employment for military and non-military for use in control totals process
	military_out = remi_ind.copy()
	military_out['military_employment'] = np.where(military_out['industry'] == '99', military_out['employment'], 0)
	military_out['non_military_employment'] = np.where(military_out['industry'] != '99', military_out['employment'], 0)
	military_out = (
		military_out[[ 'non_military_employment','military_employment','employment']]
		.sum().round(0).astype(int).to_frame(name=f'{forecast_year}').reset_index(names='variable')
	)
	region_summed = pd.concat([region_summed,military_out])
	util.save_table("region_summed", region_summed)

	non_gq_workers_ind = get_non_gq_workers(pums_person, 'industry')
	worker_to_emp_ratio_ind = get_workers_to_remi_emp_ratio(util, non_gq_workers_ind, util.get_setting('base_year'), category_col)
	remi_ind['industry'] = pd.to_numeric(remi_ind['industry'], errors='coerce')
	remi_ind = remi_ind.set_index(['county_id','industry'])['employment']
	remi_labor_force_ind = remi_ind * worker_to_emp_ratio_ind
	
	non_gq_workers_occ = get_non_gq_workers(pums_person, 'occupation')
	worker_to_emp_ratio_occ = get_workers_to_remi_emp_ratio_occ(util, non_gq_workers_occ, util.get_setting('base_year'), category_col)
	remi_emp = remi_emp.rename(columns={'occupation_code':'occupation'}).set_index(['county_id','occupation'])['employment']
	remi_labor_force_occ = remi_emp * worker_to_emp_ratio_occ

	# copy just hh to new series to calculate hhsz and workers which are based on hh
	remi_hh = remi_age['hh'].copy()
	# mutiply hh by hhsz rates by age group of head
	remi_hhsz = remi_hh * hhsz_rates
	hhsz_out = remi_hhsz.dropna().unstack().add_prefix('hhsz').groupby(['county_id']).sum()
	# multiply hhsz by workers per hhsz rates to get workers by hhsz and age group of head
	remi_workers = remi_hhsz * workers_per_hhsz_rates
	remi_workers_out = remi_workers.dropna().unstack().add_prefix('workers').groupby(['county_id']).sum()

	# aggregate age groups to county level and merge with hh, hhsz, and workers by hhsz
	out = aggregate_age_groups(remi_age[['hhpop']]).copy()
	out = out["hhpop"].unstack()
	hh_out = remi_age.groupby('county_id')['hh'].sum()
	out['hh'] = hh_out
	out = out.merge(hhsz_out, left_index=True, right_index=True, how="left")
	out = out.merge(remi_workers_out, left_index=True, right_index=True, how="left")
	# merge labor force by industry and occupation
	out_labor_force_ind = remi_labor_force_ind.unstack().add_prefix('naics_')
	out = out.merge(out_labor_force_ind, left_index=True, right_index=True, how="left")
	out_labor_force_occ = remi_labor_force_occ.unstack().add_prefix('soc_')
	out = out.merge(out_labor_force_occ, left_index=True, right_index=True, how="left")
	out = out.fillna(0).round(0).astype(int)
	# clean up and save county_controls (marginals) table for use later by popsim
	out = out.rename(columns={"hh": "num_hh"})
	out.index.name = "county_id"
	util.save_table("county_controls", out.reset_index())

def run_step(context):
	print("Generating county controls from REMI and prepared PUMS...")
	util = Util(settings_path=context["configs_dir"])
	build_remi_controls(util)
	return context
