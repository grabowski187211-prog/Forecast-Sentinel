"""Forecast Model Sentinel — guards production ML models via DataHub ML lineage.

The sentinel answers one question that ML teams cannot currently answer from a
dashboard: *this upstream thing changed — is my deployed model still valid?*

It walks DataHub's end-to-end ML lineage (training data -> features -> model ->
deployment), decides whether the change invalidates the deployed artefact, and
writes its verdict back into the catalog so the next person sees it.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
