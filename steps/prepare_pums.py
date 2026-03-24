import pandas as pd
import numpy as np
from utils import Util


def _extract_naics_2digit(value):
    if pd.isna(value):
        return pd.NA

    text = str(value).strip().upper()
    match = pd.Series([text]).str.extract(r"^(\d{2})", expand=False).iloc[0]
    if pd.notna(match):
        return int(match)

    if text.startswith("3MS"):
        return 33
    if text.startswith("4MS"):
        return 44

    return pd.NA


def get_filename(tablename, util):
    # Returns the filename for a given table from settings.yaml
    table_list = util.settings.get('pums_table_list', [])
    for table in table_list:
        if table['tablename'] == tablename:
            return table['filename']


def _build_age_labels():
    age_bins = list(range(0, 90, 5)) + [float("inf")]
    age_labels = ["ages_0_4"] + [f"ages_{i}_{i+4}" for i in range(5, 85, 5)] + ["ages_85_plus"]
    return age_bins, age_labels


def _add_county_id(pums_hh, pums_person, puma_geog_lookup):
    if "county_id" not in puma_geog_lookup.columns:
        return pums_hh, pums_person

    puma_cnty = puma_geog_lookup.groupby("PUMA")["county_id"].first().reset_index()
    pums_hh = pums_hh.merge(puma_cnty, on="PUMA", how="left")
    pums_person = pums_person.merge(puma_cnty, on="PUMA", how="left")
    pums_hh = pums_hh.loc[pums_hh["county_id"].notna()].copy()
    pums_person = pums_person.loc[pums_person["county_id"].notna()].copy()
    return pums_hh, pums_person

def prepare_pums(util):
    """
    Combine and filter PUMS data for use with other data
    """

    pums_hh = pd.read_csv(f"{util.get_data_dir()}/{get_filename('pums_hh', util)}",low_memory=False)
    pums_person = pd.read_csv(f"{util.get_data_dir()}/{get_filename('pums_person', util)}",low_memory=False)
    puma_geog_lookup = pd.read_csv(f"{util.get_data_dir()}/{get_filename('puma_geog_lookup', util)}",low_memory=False)

    # Filter for records that exist only within the geo_cross_walk
    pums_hh = pums_hh[pums_hh['PUMA'].isin(puma_geog_lookup['PUMA'])]
    pums_person = pums_person[pums_person['PUMA'].isin(puma_geog_lookup['PUMA'])]

    # Add county IDs for PUMA geography and keep only mapped records.
    pums_hh, pums_person = _add_county_id(pums_hh, pums_person, puma_geog_lookup)

    # remove records for vacant units
    pums_hh = pums_hh.loc[pums_hh['NP'] > 0].copy()

    # set group quarters flag on household and person tables
    pums_hh['gq'] = np.where(pums_hh['TYPEHUGQ'] == 1, 0, 1)
    is_gq = pums_hh.loc[pums_hh['gq'] == 1, 'SERIALNO'].values
    pums_person['gq'] = pums_person['SERIALNO'].map(pums_hh.set_index('SERIALNO')['gq'])

    age_bins, age_labels = _build_age_labels()
    pums_person['age_group'] = pd.cut(
        pums_person['AGEP'],
        bins=age_bins,
        labels=age_labels,
        right=False,
        include_lowest=True,
    )
    pums_hh['age_head_group'] = pd.cut(
        pums_hh['HHLDRAGEP'],
        bins=age_bins,
        labels=age_labels,
        right=False,
        include_lowest=True,
    )
    
    # Generate unique household ID "hhnum"
    pums_hh['hhnum'] = [i+1 for i in range(len(pums_hh))]
    pums_person['hhnum'] = 0
    pums_person['hhnum'] = pums_person['SERIALNO'].map(pums_hh.set_index('SERIALNO')['hhnum'])
    
    # Calculate household workers based on person records
    pums_person['is_worker'] = 0
    pums_person.loc[pums_person['ESR'].isin([1,2,4,5]), 'is_worker'] = 1
    worker_count = pums_person.groupby('hhnum')['is_worker'].sum().to_frame()
    pums_hh['workers'] = 0
    pums_hh.index = pums_hh.hhnum
    pums_hh.update({'workers': worker_count.is_worker})
    pums_hh['workers'] = pums_hh['workers'].clip(upper=4)

    pums_hh['hhsz'] = pums_hh['NP'].clip(upper=5)

    # Save prepared PUMS tables for downstream control generation in remi_controls.
    util.save_table('pums_person_prepared', pums_person.reset_index(drop=True))
    util.save_table('pums_households_prepared', pums_hh.reset_index(drop=True))

    # Filter to non-group-quarter records for seed tables.
    pums_hh = pums_hh.loc[pums_hh['gq'] == 0].copy()
    pums_person = pums_person.loc[pums_person['gq'] == 0].copy()

    # Combine households with workers >= 3
    #pums_hh.loc[pums_hh['worker_count'] >= 3,'worker_count'] = 3

    # we are using 2021 5 year pums to have consistent PUMs geography (2010). 
    # adjust income to 2022. 
    pums_hh['HINCP'] = pums_hh.HINCP * (pums_hh.ADJINC/1000000)
    # adjust home value
    pums_hh['VALP'] = pums_hh['VALP'] * (pums_hh.ADJHSG/1000000)

    # add occupation group codes
    occ_code_xwalk = {
        11: 1113, 13: 1113, 15: 1517, 17: 1517, 19: 19, 21: 21, 23: 23, 25: 25,
        27: 27, 29: 2931, 31: 2931, 33: 33, 35: 35, 37: 3739, 39: 3739, 41: 4143,
        43: 4143, 45: 45, 47: 47, 49: 49, 51: 51, 53: 53, 55: 55
    }
    pums_person['SOCP_2digit'] = pums_person['SOCP'].str[:2].astype(float)
    pums_person['occupation'] = pums_person['SOCP_2digit'].map(occ_code_xwalk)

    ind_code_xwalk = {
        11: 11, 21: 21, 22: 22, 23: 23, 31: 3133, 32: 3133, 33: 3133, 42: 42, 44: 4445, 45: 4445,
        48: 4849, 49: 4849, 51: 51, 52: 52, 53: 53, 54: 54, 55: 55, 56: 56, 61: 61,
        62: 62, 71: 71, 72: 72, 81: 81, 92: 92
    }
    # PyTables table format cannot serialize pandas nullable Int64 extension dtype.
    # Keep these as numpy float64 (with NaN for missing) for HDF compatibility.
    pums_person['NAICSP_2digit'] = pd.to_numeric(
        pums_person['NAICSP'].apply(_extract_naics_2digit), errors='coerce'
    ).astype('float64')
    pums_person['industry'] = pd.to_numeric(
        pums_person['NAICSP_2digit'].map(ind_code_xwalk), errors='coerce'
    ).astype('float64')

    pums_hh['region'] = 1
    util.save_table("seed_persons", pums_person)
    util.save_table("seed_households", pums_hh)

def run_step(context):
    print("Preparing PUMS data...")
    util = Util(settings_path=context['configs_dir'])
    prepare_pums(util)
    return context