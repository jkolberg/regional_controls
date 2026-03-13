import shutil
from pathlib import Path

from utils import Util


def _get_regional_controls_stem(util: Util) -> str:
	for table in util.get_setting("regional_forecast", []):
		if table.get("tablename") == "regional_controls":
			filename = table.get("filename")
			if filename:
				return Path(filename).stem

	raise ValueError(
		"Missing regional_forecast tablename=regional_controls filename in settings.yaml"
	)


def archive_outputs(util: Util):
	repo_root = Path(__file__).resolve().parent.parent
	output_dir = repo_root / "output"
	data_dir = repo_root / "data"
	archive_dir = output_dir / "archive" / _get_regional_controls_stem(util)
	archive_dir.mkdir(parents=True, exist_ok=True)

	files_to_copy = [
		(output_dir / "final_summary_county_id.csv", "final_summary_county_id.csv"),
		(output_dir / "synthetic_households.csv", "synthetic_households.csv"),
		(output_dir / "synthetic_persons.csv", "synthetic_persons.csv"),
		(data_dir / "remi_controls.csv", "remi_controls.csv"),
	]

	for source, filename in files_to_copy:
		if not source.exists():
			raise FileNotFoundError(f"Expected output file not found: {source}")

		destination = archive_dir / filename
		shutil.copy2(source, destination)
		print(f"Archived {source} to {destination}")


def run_step(context):
	print("Archiving PopulationSim output files...")
	util = Util(settings_path=context["configs_dir"])
	archive_outputs(util)
	return context
