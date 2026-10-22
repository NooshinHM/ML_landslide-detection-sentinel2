from pathlib import Path
import json
import joblib

DT_RESTORE_DIR = Path('C:\\Users\\hadid\\Ramen\\ds-ml-project-template\\artifacts\\decision_tree_handoff_20261018_222647_720637')

manifest = json.loads(
    (DT_RESTORE_DIR / "manifest.json").read_text(encoding="utf-8")
)

for group_name, group_info in manifest["groups"].items():
    print("Loading:", group_name, flush=True)
    restored_objects = joblib.load(
        DT_RESTORE_DIR / group_info["filename"]
    )
    globals().update(restored_objects)

print("Decision Tree results restored.")
print("Helper functions remain in the original notebook.")
