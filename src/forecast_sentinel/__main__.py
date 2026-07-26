"""Allow `python -m forecast_sentinel` as an alias for the `sentinel` command."""

from forecast_sentinel.cli import app

if __name__ == "__main__":
    app()
