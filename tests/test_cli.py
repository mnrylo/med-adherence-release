"""Every module imports, the command-line entry points answer --help, the shell scripts parse."""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULES = sorted(f"{p.parent.name}.{p.stem}" for p in (ROOT / "src").glob("*/*.py") if p.stem != "__init__")
MAINS = ["thesis.train_config", "thesis.sweep", "thesis.metrics", "thesis.postproc", "thesis.registry",
         "decoders.d0_reproduce", "decoders.d1_run", "decoders.d2_run", "decoders.d3_run", "decoders.a4_tuning_data",
         "embedded.phase0_predict", "embedded.phase0_eval", "embedded.phase0_quant", "embedded.phase1_bench",
         "embedded.phase2_campaign", "embedded.phase2_analysis"]


@pytest.mark.parametrize("module", MODULES)
def test_import(module):
    __import__(module)


@pytest.mark.parametrize("module", MAINS)
def test_help(module):
    r = subprocess.run([sys.executable, "-m", module, "--help"], cwd=ROOT, capture_output=True, text=True,
                       env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")})
    assert r.returncode == 0, r.stderr[-2000:]
    assert "usage" in r.stdout.lower()


@pytest.mark.parametrize("script", sorted((ROOT / "scripts").glob("*.sh")), ids=lambda p: p.name)
def test_shell_syntax(script):
    assert subprocess.run(["bash", "-n", str(script)]).returncode == 0
