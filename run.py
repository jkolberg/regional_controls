from pathlib import Path
from pypyr import pipelinerunner
import subprocess


# no command line args in this example

#----------------------------------------------------------
# Run the pypyr pipeline to prepare inputs and generate controls
#----------------------------------------------------------
configs_dir = Path(__file__).parent / "configs_pypyr"
pipelinerunner.run(f'{configs_dir}/settings', dict_in={'configs_dir': configs_dir})


#----------------------------------------------------------
# Run populationsim
#----------------------------------------------------------
popsim_configs_dir = Path(__file__).parent / "configs_popsim"
data_dir = Path(__file__).parent / "data"
output_dir = Path(__file__).parent / "output"

returncode = subprocess.call([
    ".venv/Scripts/python.exe", "-m", "populationsim", 
    '--config', str(popsim_configs_dir),
    '--data', str(data_dir),
    '--output', str(output_dir)
    ])