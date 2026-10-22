#!/usr/bin/env bash
# Pack the uncommitted RL work plus checkpoints and logs, to continue on another machine (e.g. the RTX 5090 PC).
# Everything else comes from `git clone https://github.com/ApurvK032/mira-manipulation`.
# Usage: bash rl/make_handoff.sh   ->  runs/handoff/mira_rl_handoff_<time>.tar.gz
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs/handoff
out="runs/handoff/mira_rl_handoff_$(date +%Y%m%d_%H%M).tar.gz"
tar --exclude='__pycache__' --exclude='*.pyc' -czf "$out" \
    rl docs/rl scripts/sysid_record.py \
    runs/rl runs/sysid runs/rl_*.log
echo "$out ($(du -h "$out" | cut -f1))"
tar -tzf "$out" | grep -E "(grab4|grab5|place2|place4)_ppo/model_[0-9]+\.zip" | sort -t_ -k3 -n
