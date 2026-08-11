"""Compare each mouse's next_settings in subjects.csv to the defaults."""

import csv
import json
import sys
import types
from pathlib import Path

CSV_PATH = Path(__file__).parent.parent / "data" / "village01" / "subjects.csv"
CODE_DIR = Path(__file__).parent


def load_defaults() -> dict:
    """Hacky way to not have to import village, to load
    settings from training_protocol"""
    base_mod = types.ModuleType(
        "village.custom_classes.training_protocol_base")

    class TrainingProtocolBase:
        def __init__(self):
            self.settings = types.SimpleNamespace()

    base_mod.TrainingProtocolBase = TrainingProtocolBase
    sys.modules["village"] = types.ModuleType("village")
    sys.modules["village.custom_classes"] = types.ModuleType(
        "village.custom_classes")
    sys.modules["village.custom_classes.training_protocol_base"] = base_mod

    sys.path.insert(0, str(CODE_DIR))
    from training_protocol import TrainingProtocol

    protocol = TrainingProtocol()
    protocol.default_training_settings()
    return vars(protocol.settings)


def diff_settings(defaults, settings: dict) -> list[str]:
    """return list of params whose values are not the default ones."""
    diffs = []
    for key, default in defaults.items():
        if key not in settings:
            diffs.append(f"{key}: MISSING (default {default})")
            continue
        value = settings[key]
        same = (abs(value - default) < 1e-9
                if isinstance(default, float)
                and isinstance(value, (int, float))
                else value == default)
        if not same:
            diffs.append(f"{key}: {value} (default {default})")
    return diffs


def apply_defaults(defaults: dict) -> None:
    """Overwrite every subject's next_settings with the code defaults."""
    backup = CSV_PATH.with_suffix(".csv.bak-reset-to-default")
    backup.write_bytes(CSV_PATH.read_bytes())

    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        fieldnames = reader.fieldnames
        rows = list(reader)

    for row in rows:
        row["next_settings"] = json.dumps(defaults)

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";",
                                quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Reset next_settings to defaults for {len(rows)} subjects "
          f"({', '.join(r['name'] for r in rows)}).")
    print(f"Backup of the previous file: {backup}")


if __name__ == "__main__":
    defaults = load_defaults()
    if "--apply" in sys.argv:
        apply_defaults(defaults)
    else:
        with open(CSV_PATH, newline="") as f:
            for row in csv.DictReader(f, delimiter=";"):
                print(f"=== {row['name']} ===")
                diffs = diff_settings(defaults,
                                      json.loads(row["next_settings"]))
                if diffs:
                    for d in diffs:
                        print(" ", d)
                else:
                    print("  all match defaults")
