#!/usr/bin/env python
"""Seed a small demand-forecasting ML graph into DataHub, then break it.

Why this exists: the `showcase-ecommerce` datapack is rich in datasets but thin
in `mlModel` entities, and the sentinel needs an end-to-end ML path to walk. This
emits the smallest graph that exercises the whole thing:

    raw_sales ──> feat_seasonality ──> [train_demand_forecast] ──> demand-forecast-v3
    (snowflake)    (snowflake)              (airflow job)              (mlflow)
                                                                            │
                                                                     planning-api
                                                                     (deployment)

`--break-schema` then flips `raw_sales.holiday_flag` from INT to VARCHAR — the
exact silent-invalidation case the sentinel is built to catch.

Usage:
    python scripts/seed_ml_demo.py                  # emit the graph
    python scripts/seed_ml_demo.py --break-schema   # flip the dtype
    python scripts/seed_ml_demo.py --dry-run        # build entities, emit nothing

Note: `datahub.sdk` is marked experimental by DataHub, so class signatures can
move between versions. `--dry-run` constructs every entity without touching the
network, which is the fastest way to find out if that happened.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

# datahub.sdk emits an ExperimentalWarning on import; we acknowledge it in the
# docstring above rather than printing it on every run.
warnings.filterwarnings("ignore", message=".*datahub SDK.*experimental.*")

try:
    from datahub.emitter.mcp import MetadataChangeProposalWrapper
    from datahub.metadata import schema_classes as models
    from datahub.sdk import DataHubClient
    from datahub.sdk.datajob import DataJob
    from datahub.sdk.dataset import Dataset
    from datahub.sdk.mlmodel import MLModel
    from datahub.sdk.mlmodelgroup import MLModelGroup
except ImportError:
    sys.exit(
        "The DataHub SDK is not installed. Install the optional extra:\n"
        "  uv pip install -e '.[cli]'"
    )

PLATFORM_WAREHOUSE = "snowflake"
PLATFORM_ORCHESTRATOR = "airflow"
PLATFORM_ML = "mlflow"

RAW_NAME = "sales.raw_sales"
FEATURE_NAME = "sales.feat_seasonality"
JOB_FLOW = "demand_planning"
JOB_NAME = "train_demand_forecast"
MODEL_GROUP_ID = "demand-forecast"
MODEL_ID = "demand-forecast-v3"
DEPLOYMENT_URN = (
    "urn:li:mlModelDeployment:(urn:li:dataPlatform:sagemaker,planning-api,PROD)"
)

STATUS_TAGS = (
    (
        "sentinel-reviewed",
        "Sentinel Reviewed",
        "Forecast Sentinel reviewed this model and found no blocking invalidation.",
        "#2F7D51",
    ),
    (
        "model-invalidated",
        "Model Invalidated",
        "Forecast Sentinel found an upstream change that invalidates this model.",
        "#B3261E",
    ),
    (
        "model-needs-review",
        "Model Needs Review",
        "Forecast Sentinel requires human review or more catalog evidence.",
        "#A86400",
    ),
)

# holiday_flag is the field the demo breaks. Everything else stays fixed so the
# drift report has exactly one high-severity finding to reason about.
RAW_SCHEMA_OK = [
    ("order_date", "DATE", "Date of the order."),
    ("sku_id", "VARCHAR", "Stock keeping unit identifier."),
    ("units_sold", "FLOAT", "Units sold on this date."),
    ("holiday_flag", "INT", "1 when the date is a public holiday, else 0."),
    ("region", "VARCHAR", "Sales region code."),
]
RAW_SCHEMA_BROKEN = [
    ("order_date", "DATE", "Date of the order."),
    ("sku_id", "VARCHAR", "Stock keeping unit identifier."),
    ("units_sold", "FLOAT", "Units sold on this date."),
    # The break: an upstream team switched to named holidays.
    ("holiday_flag", "VARCHAR", "Holiday name, or empty string when not a holiday."),
    ("region", "VARCHAR", "Sales region code."),
]
FEATURE_SCHEMA = [
    ("sku_id", "VARCHAR", "Join key."),
    ("week_of_year", "INT", "ISO week number."),
    ("holiday_flag", "INT", "Passed through from raw_sales; encoded at training time."),
    ("trailing_4w_mean", "FLOAT", "Trailing four-week mean of units sold."),
]


def build_client(dry_run: bool) -> DataHubClient | None:
    if dry_run:
        return None
    server = os.getenv("DATAHUB_GMS_URL", "http://localhost:8080")
    token = os.getenv("DATAHUB_GMS_TOKEN") or None
    tenant = os.getenv("DATAHUB_TENANT_URL")
    if os.getenv("DATAHUB_MODE", "selfhosted").lower() == "cloud" and tenant:
        server = tenant.rstrip("/")
        token = os.getenv("DATAHUB_TOKEN") or token
    print(f"Connecting to DataHub at {server}")
    return DataHubClient(server=server, token=token)


def upsert(client: DataHubClient | None, entity: object, label: str) -> None:
    if client is None:
        print(f"  [dry-run] would upsert {label}")
        return
    client.entities.upsert(entity)  # type: ignore[arg-type]
    print(f"  upserted {label}")


def upsert_status_tags(client: DataHubClient | None) -> None:
    """Ensure every tag used by write-back exists before the demo is checked."""
    for tag_id, name, description, colour in STATUS_TAGS:
        proposal = MetadataChangeProposalWrapper(
            entityUrn=f"urn:li:tag:{tag_id}",
            aspect=models.TagPropertiesClass(
                name=name,
                description=description,
                colorHex=colour,
            ),
        )
        if client is None:
            print(f"  [dry-run] would upsert tag {tag_id}")
        else:
            # The experimental high-level SDK has no Tag entity yet. Its
            # underlying graph client is the supported MCP emitter used by the
            # rest of acryl-datahub for aspect upserts.
            client._graph.emit_mcp(proposal)  # type: ignore[attr-defined]
            print(f"  upserted tag {tag_id}")


def seed(client: DataHubClient | None, *, broken: bool) -> None:
    upsert_status_tags(client)

    raw = Dataset(
        platform=PLATFORM_WAREHOUSE,
        name=RAW_NAME,
        description="Raw daily sales rows landed from the ERP.",
        schema=RAW_SCHEMA_BROKEN if broken else RAW_SCHEMA_OK,
    )
    upsert(client, raw, f"dataset {RAW_NAME}" + (" (broken schema)" if broken else ""))

    if broken:
        print(
            "\nholiday_flag is now VARCHAR. Run:\n"
            f"  sentinel check '{model_urn()}'"
        )
        return

    feature = Dataset(
        platform=PLATFORM_WAREHOUSE,
        name=FEATURE_NAME,
        description="Materialised seasonality features for demand forecasting.",
        schema=FEATURE_SCHEMA,
    )
    upsert(client, feature, f"dataset {FEATURE_NAME}")

    job = DataJob(
        name=JOB_NAME,
        flow_urn=f"urn:li:dataFlow:({PLATFORM_ORCHESTRATOR},{JOB_FLOW},PROD)",
        description="Nightly training run for the demand forecast model.",
    )
    upsert(client, job, f"datajob {JOB_NAME}")

    group = MLModelGroup(
        id=MODEL_GROUP_ID,
        platform=PLATFORM_ML,
        name="Demand Forecast",
        description="All versions of the SKU-level weekly demand forecast.",
    )
    upsert(client, group, f"mlModelGroup {MODEL_GROUP_ID}")

    model = MLModel(
        id=MODEL_ID,
        platform=PLATFORM_ML,
        name="Demand Forecast v3",
        description="Gradient-boosted weekly demand forecast at SKU level.",
        custom_properties={
            "framework": "lightgbm",
            "target": "units_sold",
            "horizon_weeks": "13",
        },
    )
    model.set_model_group(group.urn)
    model.add_training_job(str(job.urn))
    model.add_deployment(DEPLOYMENT_URN)
    upsert(client, model, f"mlModel {MODEL_ID}")

    if client is not None:
        # raw_sales -> feat_seasonality, and feat_seasonality -> the training job.
        client.lineage.add_lineage(upstream=raw.urn, downstream=feature.urn)
        print("  linked raw_sales -> feat_seasonality")
        client.lineage.add_datajob_lineage(datajob=job.urn, upstreams=[feature.urn])
        print("  linked feat_seasonality -> train_demand_forecast")
    else:
        print("  [dry-run] would link raw_sales -> feat_seasonality -> training job")

    print(
        "\nSeeded. Next:\n"
        f"  sentinel baseline '{model_urn()}'\n"
        "  python scripts/seed_ml_demo.py --break-schema\n"
        f"  sentinel check '{model_urn()}'"
    )


def model_urn() -> str:
    return f"urn:li:mlModel:(urn:li:dataPlatform:{PLATFORM_ML},{MODEL_ID},PROD)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--break-schema",
        action="store_true",
        help="Flip raw_sales.holiday_flag from INT to VARCHAR.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Construct every entity but emit nothing. Verifies SDK compatibility.",
    )
    args = parser.parse_args()

    client = build_client(args.dry_run)
    seed(client, broken=args.break_schema)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
