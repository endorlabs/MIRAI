#!/bin/bash
#Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Use this script to update the seed summary store in mirai/binaries/summary_store.tar

# Exit immediately if a command exits with a non-zero status.
set -euo pipefail

cargo build --locked

target_dir=$(cargo metadata --format-version=1 --no-deps | python3 -c 'import json, sys; print(json.load(sys.stdin)["target_directory"])')
artifacts=$(mktemp)
trap 'rm -f "$artifacts"' EXIT

# build the mirai-standard-contracts crate
touch standard_contracts/src/lib.rs
cargo build --locked --lib -p mirai-standard-contracts
touch standard_contracts/src/lib.rs
RUSTC_WRAPPER="$target_dir/debug/mirai" RUST_BACKTRACE=1 MIRAI_LOG=warn MIRAI_START_FRESH=true MIRAI_SHARE_PERSISTENT_STORE=true MIRAI_FLAGS="--diag=paranoid" cargo build --locked --lib -p mirai-standard-contracts --message-format=json > "$artifacts"

# collect the summary store into a tar file
summary_dir=$(python3 - "$artifacts" <<'PY'
import json
import pathlib
import sys

for line in pathlib.Path(sys.argv[1]).read_text().splitlines():
    artifact = json.loads(line)
    if artifact.get("reason") == "compiler-artifact" and artifact["target"]["name"] == "mirai_standard_contracts":
        for filename in artifact["filenames"]:
            if filename.endswith(".rmeta"):
                print(pathlib.Path(filename).parent)
                sys.exit(0)
raise SystemExit("standard contracts metadata artifact was not produced")
PY
)
tar -C "$summary_dir" -c -f binaries/summary_store.tar .summary_store.sled
