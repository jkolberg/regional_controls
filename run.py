from pathlib import Path
from pypyr import pipelinerunner


# no command line args in this example

#----------------------------------------------------------
# Run the pypyr pipeline to prepare inputs and generate controls
#----------------------------------------------------------
configs_dir = Path(__file__).parent / "configs_pypyr"
pipelinerunner.run(f'{configs_dir}/settings', dict_in={'configs_dir': configs_dir})