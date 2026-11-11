"""Provenance records for the launch-site / launch-vehicle data, so reports can
say where every number came from and how much to trust it.

    from orbitopt.launch_constraints import LaunchConstraints
    lc = LaunchConstraints.from_names("falcon-9", 4000.0, "kourou")
    print(format_data_sources(lc.data_sources()))

Provenance vocabulary (see docs/launch-data-sources.md):

* ``official``        -- published by the launch provider / agency.
* ``public_estimate`` -- secondary reference, digitized curve, mission-history
  reconstruction ... approximate; never present it as fact.
* ``derived``         -- computed by orbitopt from other fields (its own
  confidence is that of its inputs, which are listed separately).
* ``illustrative``    -- authored for demonstration (Flight Lab scenarios,
  synthetic runs): plausible numbers, no claim to match any real vehicle.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

PROVENANCES = ("official", "public_estimate", "derived", "illustrative")


@dataclass(frozen=True)
class DataSource:
    """Where one data item (e.g. ``falcon-9 c3_curve``) came from."""

    item: str
    provenance: str
    source: str
    accuracy: str = ""
    as_of: str = ""

    @property
    def is_estimate(self) -> bool:
        return self.provenance == "public_estimate"

    def to_dict(self) -> dict:
        return asdict(self)


def overall_confidence(sources: Iterable[DataSource]) -> str:
    """``"public_estimate"`` if any item is an estimate, else ``"official"``
    (``derived`` items inherit the confidence of their listed inputs)."""
    return "public_estimate" if any(s.is_estimate for s in sources) else "official"


def data_sources_report(sources: Sequence[DataSource]) -> dict:
    """JSON-ready summary: overall confidence + one record per item."""
    return {
        "confidence": overall_confidence(sources),
        "items": [s.to_dict() for s in sources],
    }


def format_data_sources(sources: Sequence[DataSource]) -> str:
    """Human-readable block listing each item's provenance, accuracy, date
    and source, ending with a caveat when estimates are involved."""
    if not sources:
        return "Data sources: none"
    lines = [f"Data sources (overall confidence: {overall_confidence(sources)}):"]
    for s in sources:
        meta = ", ".join(x for x in (s.provenance, s.accuracy, f"as of {s.as_of}" if s.as_of else "") if x)
        lines.append(f"  - {s.item} [{meta}]")
        lines.append(f"      {s.source}")
    if any(s.is_estimate for s in sources):
        lines.append("  Note: public_estimate values are approximate reconstructions, not official "
                     "figures -- see docs/launch-data-sources.md.")
    return "\n".join(lines)
