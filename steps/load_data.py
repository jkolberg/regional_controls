import shutil
import re
from pathlib import Path

import pandas as pd
from utils import Util


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

    util.save_table('regional_controls', remi)

def run_step(context):
    # pypyr step to run load_data.py
    print("Loading data into HDF5 store...")
    util = Util(settings_path=context['configs_dir'])
    get_missing_tables(util)
    load_tables(util)
    load_regional_controls_table(util)
    return context