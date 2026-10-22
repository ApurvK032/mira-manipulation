#!/usr/bin/env bash
# Pack everything another machine needs to continue from this checkout: every file changed or added since the last
# commit (tracked or not, minus ignored files), the trained policies (each run's final save, plus the checkpoints the
# docs name), progress logs and evaluation logs, the real-arm system-ID recordings and the arm configuration backups.
# Extract at the root of a clone of the same commit. Usage: bash rl/make_sync_bundle.sh -> runs/handoff/mira_sync_<time>.tar.gz
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs/handoff
out="runs/handoff/mira_sync_$(date +%Y%m%d_%H%M).tar.gz"
list=$(mktemp)
{
    git diff --name-only HEAD
    git ls-files --others --exclude-standard
} | grep -v -E '^MUJOCO_LOG\.TXT$' | sort -u > "$list"
# runs/ is ignored by git: add what is needed explicitly
find runs/rl -maxdepth 2 \( -name 'model.zip' -o -name 'vecnormalize.pkl' -o -name 'progress.log' -o -name 'config.json' \) >> "$list"
for f in runs/rl/place2_ppo/model_42000048.zip runs/rl/place2_ppo/vecnormalize_42000048.pkl \
         runs/rl/grab4_ppo/model_14000016.zip runs/rl/grab4_ppo/vecnormalize_14000016.pkl; do
    [ -f "$f" ] && echo "$f" >> "$list"
done
find runs/rl/lageval runs/rl/reaim runs/rl/robust runs/rl/stream runs/rl/upgrade_sim runs/sysid runs/arm_backup runs/deploy \
     -type f 2>/dev/null >> "$list" || true
ls runs/rl_*.log >> "$list" 2>/dev/null || true
tar --exclude='__pycache__' --exclude='*.pyc' -czf "$out" -T "$list"
echo "$out ($(du -h "$out" | cut -f1), $(wc -l < "$list") files, base commit $(git rev-parse --short HEAD))"
rm -f "$list"
