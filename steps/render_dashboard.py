import subprocess
import sys
from pathlib import Path


def render_dashboard():
	repo_root = Path(__file__).resolve().parent.parent
	qmd_path = repo_root / "dashboard.qmd"
	docs_dir = repo_root / "docs"

	if not qmd_path.exists():
		raise FileNotFoundError(f"Dashboard source not found: {qmd_path}")

	result = subprocess.run(
		["quarto", "render", str(qmd_path), "--output-dir", str(docs_dir)],
		cwd=str(repo_root),
		capture_output=True,
		text=True,
	)

	if result.returncode != 0:
		print(result.stdout)
		print(result.stderr, file=sys.stderr)
		raise RuntimeError(f"Quarto render failed with exit code {result.returncode}")

	print(f"Dashboard rendered to {docs_dir / 'index.html'}")


def run_step(context):
	print("Rendering Quarto dashboard...")
	render_dashboard()
	return context
