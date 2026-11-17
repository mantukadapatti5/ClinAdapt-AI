"""Verify the local dataset bundles against the parsers used by the API.

Run from backend/:  python verify_datasets.py
Exit code 0 when every check passes, 1 otherwise.
"""
import json
import sys

from app import datasets


def main() -> int:
    report = datasets.verify()
    print(json.dumps(report, indent=2))
    status = "OK" if report["ok"] else "FAILED"
    print(f"\n{status}: {len(report['errors'])} error(s), {len(report['warnings'])} warning(s), "
          f"{len(report['notes'])} note(s)", file=sys.stderr)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
