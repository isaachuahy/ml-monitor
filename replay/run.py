"""Generate a static replay snapshot after both isolated Compose runs pass."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import tempfile
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "replay" / "public" / "replay.json"


def compose(project, image, *args, capture=False):
    return subprocess.run(
        ["docker", "compose", "--env-file", os.devnull,
         "-f", "docker-compose.yml", "-f", "replay/compose.yml", "-p", project, *args],
        cwd=ROOT,
        env={
            **os.environ,
            "POSTGRES_USER": "replay",
            "POSTGRES_PASSWORD": "replay",
            "POSTGRES_DB": "replay",
            "DATABASE_URL": "postgresql://replay:replay@db:5432/replay",
            "DISCORD_WEBHOOK_URL": "",
            "REPLAY_IMAGE": image,
        },
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    )


def run_scenario(project, image, scenario):
    run_worker = ("run", "--rm", "-T", "--no-deps", "--pull", "never", "eval_worker")
    try:
        compose(project, image, "up", "--wait", "--wait-timeout", "90", "db")
        compose(project, image, *run_worker, "bootstrap")
        compose(project, image, "up", "--no-build", "--wait", "--wait-timeout", "90", "api")
        result = compose(project, image, *run_worker, scenario, capture=True)
        return json.loads(result.stdout)
    finally:
        compose(project, image, "down", "--volumes")


def verify_replay():
    """Return the verified reference, model metadata, and results for both scenarios."""
    run_id = uuid4().hex
    prefix = f"ml-replay-{run_id}"
    image = f"ml-monitor-replay:{run_id}"
    compose(f"{prefix}-baseline", image, "build", "api")
    try:
        baseline = run_scenario(f"{prefix}-baseline", image, "baseline")
        shifted = run_scenario(f"{prefix}-income-shift", image, "income_shift")
    finally:
        subprocess.run(["docker", "image", "rm", image], check=False)

    baseline_result = baseline.pop("result")
    shifted_result = shifted.pop("result")
    if baseline != shifted:
        raise RuntimeError("Scenarios used different model artifacts or reference data")
    return {
        "schema_version": 1,
        "run_id": run_id,
        "synthetic": True,
        "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **baseline,
        "scenarios": {"baseline": baseline_result, "income_shift": shifted_result},
    }


def write_snapshot(output, snapshot):
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                     prefix=".replay-", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(snapshot, handle, allow_nan=False)
            handle.write("\n")
            handle.close()
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    snapshot = verify_replay()
    write_snapshot(args.output, snapshot)
    for name, result in snapshot["scenarios"].items():
        decision = "drift detected" if result["drift_detected"] else "no drift detected"
        print(f"{name}: {result['sample_count']} persisted predictions, "
              f"{decision} (p={result['ks_p_value']:.8g})")
    print(f"Verified replay saved to {args.output} ({snapshot['run_id']})")


if __name__ == "__main__":
    main()
