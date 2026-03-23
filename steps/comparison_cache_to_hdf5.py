import subprocess
from pathlib import Path
from utils import Util

def get_cache_dir(util):
    # Support both legacy and current setting names.
    cache_dir = util.get_setting('urbansim_cache_dir') or util.get_setting('urbansim_baseyear_cache')

    # check to make sure the cache is accessible
    if not Path(cache_dir).exists():
        raise ConnectionError(
            f"Unable to access network path: {cache_dir}. Make sure you are connected to the PSRC VPN or have access to the network drive."
        )
    
    return cache_dir

def get_comparison_controls_store(util):
    # get the output HDF path for comparison controls from settings.yaml
    hdf_path = util.get_setting('comparison_controls_store')
    if not hdf_path:
        raise ValueError("comparison_controls_store not found in settings.yaml. Please add comparison_controls_store to your settings.yaml file.")

    hdf_path = Path(hdf_path)
    if hdf_path.is_absolute():
        return hdf_path
    return Path(__file__).parent.parent / hdf_path

def delete_existing_hdf_store(hdf_path):
    if hdf_path.exists():
        print(f"Deleting existing HDF store at {hdf_path}...")
        hdf_path.unlink()

def run_step(context):
    # This step is for running the existing cache to HDF5 conversion script
    # which is a separate Python script that we want to run as part of the pipeline
    print("Running existing cache to HDF5 conversion...")
    util = Util(settings_path=context['configs_dir'])
    cache_dir = get_cache_dir(util)
    hdf_path = get_comparison_controls_store(util)
    delete_existing_hdf_store(hdf_path)

    if not cache_dir:
        if hdf_path.exists():
            print(f"Skipping cache conversion; using existing HDF store at {hdf_path}")
            return context
        raise ValueError(
            "urbansim_cache_dir or urbansim_baseyear_cache not found in settings.yaml, "
            "and comparison_controls_store does not exist yet."
        )

    script_path = Path(__file__).parent.parent / "utils" / "cache_to_hdf5.py"

    returncode = subprocess.call([
        ".venv/Scripts/python.exe", str(script_path),
        str(cache_dir), str(hdf_path)
        ])
    if returncode != 0:
        raise RuntimeError(f"cache_to_hdf5.py failed with exit code {returncode}")
    
    return context