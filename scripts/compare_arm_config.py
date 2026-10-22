"""Compare two arm-controller configuration backups (scripts/backup_arm_config.py), e.g. before and after a firmware
upgrade, to confirm the arm kept its settings and its factory calibration (joint_characteristics).

Usage:  python -m scripts.compare_arm_config <before.yaml> <after.yaml>
"""

import argparse

import yaml


def flatten(x, prefix=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from flatten(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from flatten(v, f"{prefix}[{i}]")
    else:
        yield prefix, x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("before")
    ap.add_argument("after")
    args = ap.parse_args()
    a = dict(flatten(yaml.safe_load(open(args.before))))
    b = dict(flatten(yaml.safe_load(open(args.after))))
    changed = [k for k in a if k in b and a[k] != b[k]]
    removed = [k for k in a if k not in b]
    added = [k for k in b if k not in a]
    calib = [k for k in changed + removed if k.startswith("joint_characteristics")]
    print(f"{len(a)} fields before, {len(b)} after: {len(changed)} changed, {len(removed)} removed, {len(added)} added")
    print("factory calibration (joint_characteristics): " + ("UNCHANGED" if not calib else f"{len(calib)} fields CHANGED or REMOVED"))
    for k in calib:
        print(f"  {k}: {a.get(k)} -> {b.get(k, '(removed)')}")
    for k in changed:
        if not k.startswith("joint_characteristics"):
            print(f"changed  {k}: {a[k]} -> {b[k]}")
    for k in removed:
        if not k.startswith("joint_characteristics"):
            print(f"removed  {k}")
    for k in added:
        print(f"added    {k}: {b[k]}")


if __name__ == "__main__":
    main()
