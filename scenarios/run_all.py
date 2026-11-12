"""Regenerate every results/*.json (and results/index.json) in one go.

    .venv/bin/python -m scenarios.run_all
"""
import importlib
import time

MODULES = ["passive_flight", "analysis", "attitude_control", "los_path", "observers", "robustness"]


def main():
    t_all = time.perf_counter()
    for name in MODULES:
        t0 = time.perf_counter()
        print(f"=== scenarios.{name}")
        importlib.import_module(f"scenarios.{name}").main()
        print(f"=== scenarios.{name}: {time.perf_counter() - t0:.1f} s\n")
    print(f"all scenarios done in {time.perf_counter() - t_all:.1f} s")


if __name__ == "__main__":
    main()
