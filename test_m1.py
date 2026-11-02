"""M1 smoke test: official retail environment + tools + real database.

Run:
    .venv\\Scripts\\python.exe test_m1.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "code"))

from harness.tau2_adapter import build_retail_env, tool_catalog  # noqa: E402


def main() -> None:
    env = build_retail_env()
    catalog = tool_catalog(env)

    print("tool_count:", len(catalog))
    print("tool_names:", [entry["function"]["name"] for entry in catalog])
    print("db_stats:", env.tools.db.get_statistics())

    # #W6067464 is the order used by task_0's ground-truth action.
    order = env.tools.db.orders["#W6067464"]
    print("order:", order.model_dump())
    print("user:", env.tools.db.users[order.user_id].model_dump())
    print("db_hash:", env.get_db_hash())


if __name__ == "__main__":
    main()
