#!/usr/bin/env bash
# Generate skeletons in chunks and commit + push them after each chunk, so an unattended run
# (e.g., in a cloud session) loses nothing if the machine goes away. Rerunning resumes:
# finished batches are skipped.
#
#   bash src/modeling/scripts/generate_and_push.sh [OUT] [BATCHES] [WORKERS] [BRANCH]
#   defaults:                                      data/full 250 3 data/skeletons-v1
#
# Generated data is gitignored on main, so it is committed with -f on its own branch.
set -uo pipefail

OUT=${1:-data/full}
BATCHES=${2:-250}
WORKERS=${3:-3}
BRANCH=${4:-data/skeletons-v1}
CHUNK=${CHUNK:-15}

# Continue the data branch if it exists on the remote, else start it from the current commit.
if git fetch -q origin "$BRANCH" 2>/dev/null; then
  git checkout -q -B "$BRANCH" "origin/$BRANCH"
else
  git checkout -q -B "$BRANCH"
fi

for (( n = CHUNK; ; n += CHUNK )); do
  (( n > BATCHES )) && n=$BATCHES
  uv run python -m modeling.data.skeletons.generate --out "$OUT" --batches "$n" \
    --workers "$WORKERS"
  done_count=$(ls "$OUT/skeletons" 2>/dev/null | wc -l)
  git add -f "$OUT/skeletons" "$OUT/raw"
  git commit -q -m "skeletons: $done_count/$BATCHES batches

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" || echo "nothing new to commit"
  git push -q origin "$BRANCH" || echo "push failed; will retry after the next chunk"
  echo "== $done_count/$BATCHES batches done =="
  (( n >= BATCHES )) && break
done

missing=$(( BATCHES - $(ls "$OUT/skeletons" | wc -l) ))
if (( missing > 0 )); then
  echo "$missing batches failed; rerun this script to retry them"
  exit 1
fi
echo "all $BATCHES batches done"
