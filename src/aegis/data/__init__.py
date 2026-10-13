"""Aegis data flywheel (minimal honest slice).

Offline export of ``outputs/<run_id>/`` NDJSON into a training-ready
``datasets/`` layout. State+action only — no ``videos/`` dir until
``telemetry/image_logger.py`` frames land in RunLogger (trigger T2 in
``docs/data-pipeline-decision.md``).

Parquet when pandas+pyarrow are present, else ``.jsonl`` fallback so the
command works on a bare checkout. See ``validate.py`` + ``aegis data-check``.
"""

from __future__ import annotations
