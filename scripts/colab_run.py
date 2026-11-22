"""Colab entry point. See scripts/cloud_run.py."""

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).with_name("cloud_run.py")), run_name="__main__")
