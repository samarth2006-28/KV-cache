"""Audit recorded fractions without modifying or silently filtering results.csv."""
import csv
import math


def audit_rows(rows, tolerance_pp=1.0, units="fraction"):
    if not math.isfinite(tolerance_pp) or tolerance_pp < 0:
        raise ValueError("tolerance_pp must be finite and nonnegative")
    if units not in ("fraction", "percent"):
        raise ValueError("units must be fraction or percent")
    factor = 100.0 if units == "fraction" else 1.0
    report = []
    for number, row in enumerate(rows, 2):
        reason = ""
        difference = None
        try:
            target = float(row["budget_target"])
            measured = float(row["budget_measured"])
            if not math.isfinite(target) or not math.isfinite(measured):
                raise ValueError("nonfinite budget")
            if not 0 <= target <= (1 if units == "fraction" else 100) or measured < 0:
                raise ValueError("budget out of range for selected units")
            difference = abs(measured-target)*factor
            if difference > tolerance_pp + 1e-10:
                reason = "measured budget outside tolerance"
        except (KeyError, TypeError, ValueError) as exc:
            reason = f"invalid budget fields: {exc}"
        report.append({"csv_line": number, "method": row.get("method", ""),
                       "eligible": not reason, "difference_pp": difference,
                       "reason": reason})
    return report


def read_audit(path, **kwargs):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"method", "budget_target", "budget_measured"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV must contain {sorted(required)}")
        rows = list(reader)
    if not rows:
        raise ValueError("CSV has no runs; an empty matrix is not a passed audit")
    return audit_rows(rows, **kwargs)


def main():
    import argparse
    import json
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv")
    p.add_argument("--units", choices=["fraction", "percent"], default="fraction")
    p.add_argument("--tolerance-pp", type=float, default=1.0)
    args = p.parse_args()
    try:
        report = read_audit(args.csv, units=args.units, tolerance_pp=args.tolerance_pp)
    except (OSError, ValueError) as exc:
        p.exit(2, f"Audit failed: {exc}\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if all(row["eligible"] for row in report) else 1


if __name__ == "__main__":
    raise SystemExit(main())
