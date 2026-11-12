"""Run the repo's E2a (micro_conformal), E2b (learned_gate) and APS/RAPS scripts on a staged dir.

Usage: python conformal_suite.py <stage-dir>
Outputs land in the stage dir: micro_conformal_results.json, learned_gate_results.json,
aps_raps_robustness.json.
"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def main():
    stage = Path(sys.argv[1]).resolve()
    for script in ("micro_conformal.py", "learned_gate.py"):
        subprocess.run([sys.executable, str(REPO / "experiments" / script), "--results-dir", str(stage)],
                       cwd=REPO, check=True, stdout=subprocess.DEVNULL)
    code = ("import sys; sys.path.insert(0, %r); sys.argv=['aps'];"
            "import aps_raps_robustness as a; from pathlib import Path; a.RESULTS = Path(%r); a.main()"
            % (str(REPO / "experiments"), str(stage)))
    subprocess.run([sys.executable, "-c", code], cwd=REPO, check=True, stdout=subprocess.DEVNULL)
    print("done", stage)


if __name__ == "__main__":
    main()
