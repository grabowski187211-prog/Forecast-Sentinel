#!/usr/bin/env bash
# Bring up a local DataHub and load enough metadata to demo the sentinel.
#
#   ./scripts/bootstrap_datahub.sh
#
# Requires Docker (>= 2 CPUs, 8GB RAM, 2GB swap, 13GB disk allocated) and the
# DataHub CLI. Quickstart is a development deployment only — default credentials,
# all ports exposed, no horizontal scaling.
set -euo pipefail

info()  { printf '\033[36m==>\033[0m %s\n' "$*"; }
warn()  { printf '\033[33mwarn:\033[0m %s\n' "$*"; }
fail()  { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

command -v docker >/dev/null 2>&1 || fail \
  "Docker not found. Install Docker Desktop, start the engine, and allocate at
   least 2 CPUs / 8GB RAM / 13GB disk before re-running."

docker info >/dev/null 2>&1 || fail "Docker is installed but not running. Start it and retry."

if command -v datahub >/dev/null 2>&1; then
  DATAHUB=datahub
elif [ -x .venv/bin/datahub ]; then
  DATAHUB=.venv/bin/datahub
else
  fail "DataHub CLI not found. Install it with one of:
   brew install datahub-project/tap/datahub
   uv pip install -e '.[cli]'"
fi

info "DataHub CLI: $($DATAHUB version 2>&1 | head -1)"

info "Starting DataHub (first run pulls several GB of images — expect 5-15 min)"
$DATAHUB docker quickstart

info "Configuring the CLI against the local instance"
$DATAHUB init --username datahub --password datahub

info "Loading the showcase-ecommerce datapack (~1,050 entities with lineage)"
if ! $DATAHUB datapack load showcase-ecommerce; then
  warn "Datapack load failed. The instance is still usable; the ML demo graph
        below does not depend on it."
fi

info "Seeding the demand-forecasting ML lineage graph"
if [ -x .venv/bin/python ]; then
  .venv/bin/python scripts/seed_ml_demo.py
else
  python scripts/seed_ml_demo.py
fi

cat <<'EOF'

Ready.
  UI:   http://localhost:9002   (datahub / datahub)
  GMS:  http://localhost:8080

Next:
  sentinel doctor
  sentinel models
  sentinel baseline "urn:li:mlModel:(urn:li:dataPlatform:mlflow,demand-forecast-v3,PROD)"
  python scripts/seed_ml_demo.py --break-schema
  sentinel check "urn:li:mlModel:(urn:li:dataPlatform:mlflow,demand-forecast-v3,PROD)"

To stop:  datahub docker quickstart --stop
EOF
