import shutil
import re
from pathlib import Path

import pandas as pd
from utils import Util


def _get_input_filename(util, tablename):
    for table in util.get_table_list():
        if table.get('tablename') == tablename:
            return table.get('filename')
    return None


def _normalize_industry_text(value):
    text = str(value).strip().lower()
    text = text.replace("n.e.c.", "")
    text = text.replace("not elsewhere classified", "")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _build_remi_industry_label_map(util):
    configured_filename = _get_input_filename(util, 'industry_crosswalk')
    if not configured_filename:
        return {}

    crosswalk_path = Path(util.get_data_dir()) / configured_filename
    if not crosswalk_path.exists():
        raise FileNotFoundError(
            f"Configured industry crosswalk not found: {crosswalk_path}. Check configs_pypyr/settings.yaml input_table_list."
        )

    industry_crosswalk = pd.read_csv(crosswalk_path)
    remi_col = 'remi_industry' if 'remi_industry' in industry_crosswalk.columns else 'industry_group_2nd_table'
    industry_col = 'industry' if 'industry' in industry_crosswalk.columns else 'industry_code'
    if remi_col not in industry_crosswalk.columns or industry_col not in industry_crosswalk.columns:
        raise KeyError(
            "industry_crosswalk must include remi_industry (or industry_group_2nd_table) and industry (or industry_code)."
        )

    label_map = (
        industry_crosswalk[[remi_col, industry_col]]
        .dropna()
        .assign(
            _key=lambda df: df[remi_col].apply(_normalize_industry_text),
            _val=lambda df: "naics_" + df[industry_col].astype(str).str.strip(),
        )
        .set_index('_key')['_val']
        .to_dict()
    )

    return label_map


def _apply_industry_crosswalk_labels(remi, util):
    category_col = 'category' if 'category' in remi.columns else 'Category'
    label_map = _build_remi_industry_label_map(util)
    if not label_map:
        return remi

    def map_category(value):
        text = str(value)
        match = re.search(r"^\s*Employment(?:\s+by\s+Major\s+Industry)?\s*-\s*(.*)$", text, flags=re.IGNORECASE)
        if not match:
            return value

        key = _normalize_industry_text(match.group(1))
        mapped = label_map.get(key)
        return mapped if mapped else value

    remi[category_col] = remi[category_col].apply(map_category)
    return remi


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


def _regional_forecast_filename(util, tablename):
    for table in util.get_setting('regional_forecast', []):
        if table.get('tablename') == tablename:
            return table.get('filename')
    return None


def _copy_if_missing(util, filename):
    data_path = Path(util.get_data_dir()) / filename
    if data_path.exists():
        return

    forecasts_dir = util.get_setting('regional_forecasts_dir')
    if not forecasts_dir:
        print(f"Missing file in data dir and no regional_forecasts_dir configured: {filename}")
        return

    source_path = Path(forecasts_dir) / filename
    if source_path.exists():
        print(f"Copying {filename} from {source_path} to {data_path}")
        shutil.copy(source_path, data_path)
    else:
        print(f"Missing table file: {filename} (not found in data dir or {forecasts_dir})")


def get_missing_tables(util):
    # check for any missing csv tables in input_table_list and fetch from regional_forecasts_dir
    for table in util.get_table_list():
        _copy_if_missing(util, table['filename'])

    # ensure REMI workbook configured under regional_forecast is present in data dir
    remi_filename = _regional_forecast_filename(util, 'regional_controls')
    if remi_filename:
        _copy_if_missing(util, remi_filename)

def load_tables(util):
    # Creates an HDF5 file and loads tables into it
    table_list = util.get_table_list()
    for table in table_list:
        print(f"Loading table: {table['tablename']} from file: {table['filename']}")
        df = pd.read_csv(f"{util.get_data_dir()}/{table['filename']}",low_memory=False)
        
        # fill nan values
        df = util.fill_nan_values(df)
        
        # create block_group_id only when geographic columns are present
        if {'state', 'county', 'tract', 'block group'}.issubset(df.columns):
            df = util.create_full_block_group_id(df)
        elif util.block_group_id_exists(df):
            df = util.convert_col_to_int64(df, 'block_group_id')
        
        # save table to HDF5 store
        util.save_table(table['tablename'], df)


def load_regional_controls_table(util):
    remi_filename = _regional_forecast_filename(util, 'regional_controls')
    if not remi_filename:
        raise ValueError(
            "Missing regional_forecast tablename=regional_controls in settings.yaml"
        )

    remi_path = Path(util.get_data_dir()) / remi_filename
    if not remi_path.exists():
        raise FileNotFoundError(
            f"Configured REMI workbook not found: {remi_path}. "
            "Check regional_forecast and regional_forecasts_dir in settings.yaml."
        )

    remi = pd.read_excel(remi_path, skiprows=5)
    county_map = util.get_setting('county_map')
    if not county_map:
        raise KeyError("Missing county_map in configs_pypyr/settings.yaml")

    remi['county_id'] = remi['Region'].map(county_map)
    remi = remi.loc[remi['county_id'].notna()].copy()
    remi['county_id'] = remi['county_id'].astype(int)
    remi['Category'] = remi['Category'].apply(_normalize_remi_age_category)
    remi = _apply_industry_crosswalk_labels(remi, util)

    util.save_table('regional_controls', remi)

def run_step(context):
    # pypyr step to run load_data.py
    print("Loading data into HDF5 store...")
    util = Util(settings_path=context['configs_dir'])
    get_missing_tables(util)
    load_tables(util)
    load_regional_controls_table(util)
    return context