import subprocess
from pathlib import Path


def run_step(context):
    popsim_configs_dir = Path(__file__).parent.parent / "configs_popsim"
    data_dir = Path(__file__).parent.parent / "data"
    output_dir = Path(__file__).parent.parent / "output"

    returncode = subprocess.call([
        ".venv/Scripts/python.exe", "-m", "populationsim", 
        '--config', str(popsim_configs_dir),
        '--data', str(data_dir),
        '--output', str(output_dir)
        ])
    
    return context