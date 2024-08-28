"""Replace the website placeholders with website domains from env_config
Generate the test data"""

import os
import json


def main() -> None:
    with open("test.raw.json", "r") as f:
        raw = f.read()
    raw = raw.replace("__GITLAB__", os.environ.get("GITLAB"))
    raw = raw.replace("__REDDIT__", os.environ.get("REDDIT"))
    raw = raw.replace("__SHOPPING__", os.environ.get("SHOPPING"))
    raw = raw.replace("__SHOPPING_ADMIN__", os.environ.get("SHOPPING_ADMIN"))
    raw = raw.replace("__WIKIPEDIA__", os.environ.get("WIKIPEDIA"))
    raw = raw.replace("__MAP__", os.environ.get("MAP"))
    with open("test.json", "w") as f:
        f.write(raw)
    # split to multiple files
    data = json.loads(raw)
    for idx, item in enumerate(data):
        with open(f"{idx}.json", "w") as f:
            json.dump(item, f, indent=2)


if __name__ == "__main__":
    main()