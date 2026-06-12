#!/bin/bash
set -euo pipefail

SRC_ROOT=/project2/ll_774_951/midterm

n="${1:?usage: $0 N|all OUT_MANIFEST}"
manifest="${2:?usage: $0 N|all OUT_MANIFEST}"

mkdir -p "$(dirname "$manifest")"

if [ "$n" = "all" ]; then
  find "$SRC_ROOT" -type f -name '*.csv' | sort > "$manifest"
else
  find "$SRC_ROOT" -type f -name '*.csv' | sort | head -n "$n" > "$manifest"
fi

echo "Wrote $(wc -l < "$manifest") files to $manifest"
