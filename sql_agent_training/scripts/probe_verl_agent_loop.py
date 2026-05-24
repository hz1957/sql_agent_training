"""Probe the installed VERL Agent Loop API."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sql_agent_training.train.verl_agent_loop_adapter import describe_verl_agent_loop_api


def main() -> None:
    try:
        description = describe_verl_agent_loop_api()
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        raise SystemExit(1)

    print(json.dumps({"ok": True, **description}, indent=2))


if __name__ == "__main__":
    main()
