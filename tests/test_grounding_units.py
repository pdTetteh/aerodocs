"""Regression tests for aviation units, identifiers, and date/modal edge cases."""

import pytest

from app.generation.grounding import semantic_compatible, semantic_signature

TORQUE = "Torque the actuator rod end jam nut to "


@pytest.mark.parametrize(
    ("claim", "evidence", "expected"),
    [
        # Torque: different dimensions or magnitudes must never be accepted.
        (TORQUE + "80 N·m.", TORQUE + "80 lbf-in.", False),
        (TORQUE + "80 in.", TORQUE + "80 in-lb.", False),
        (TORQUE + "80 ft-lb.", TORQUE + "80 in-lb.", False),
        (TORQUE + "8 N·m.", TORQUE + "80 in-lb.", False),
        # Torque: ordinary equivalent statements are accepted.
        (TORQUE + "9 N·m.", TORQUE + "80 in-lb.", True),
        (TORQUE + "80 Nm.", TORQUE + "80 N·m.", True),
        (TORQUE + "80 in-lb.", TORQUE + "80 inch-pounds.", True),
        (TORQUE + "80 in-lb.", TORQUE + "80 lbf-in.", True),
        (TORQUE + "6.7 ft-lb.", TORQUE + "80 in-lb.", True),
        # Pressure.
        ("Pressure is 689 kPa.", "Pressure is 100 psi.", True),
        ("Pressure is 100 psi.", "Pressure is 0.69 MPa.", True),
        ("Pressure is 100 kPa.", "Pressure is 100 MPa.", False),
        ("Pressure is 100 psi.", "Pressure is 6.9 MPa.", False),
        # Temperature.
        ("Temperature is 248 °F.", "Temperature is 120 °C.", True),
        ("Temperature is 250 °F.", "Temperature is 120 °C.", False),
        # Intervals: hours and cycles are different things.
        ("Inspect every 600 flight hours.", "Inspect every 600 flight cycles.", False),
        ("Inspect every 600 FH.", "Inspect every 600 flight hours.", True),
        ("Inspect every 600 flight hours.", "Inspect every 800 flight hours.", False),
    ],
)
def test_units_are_compared_by_dimension_and_value(
    claim: str, evidence: str, expected: bool
) -> None:
    assert semantic_compatible(claim, evidence) is expected


@pytest.mark.parametrize(
    ("claim", "evidence", "expected"),
    [
        ("Replace the NAS1351-3-8 screw.", "Replace the NAS1351-3-8 screw.", True),
        ("Replace the NAS1351-3-8 screw.", "Replace the NAS1351-3-6 screw.", False),
        ("Install bolt AN3-5A.", "Install bolt AN3-7A.", False),
        ("Comply with AD 2023-15-04.", "Comply with AD 2023-15-04.", True),
        ("Comply with AD 2023-15-04.", "Comply with AD 2022-15-04.", False),
        ("See task 32-31-00 for details.", "See task 32-31-00 for details.", True),
        ("See task 32-31-00 for details.", "See task 32-41-00 for details.", False),
    ],
)
def test_part_ad_and_ata_identifiers_must_match_exactly(
    claim: str, evidence: str, expected: bool
) -> None:
    assert semantic_compatible(claim, evidence) is expected


def test_identifiers_are_not_parsed_as_quantities() -> None:
    signature = semantic_signature("Replace the NAS1351-3-8 screw per AD 2023-15-04.")

    assert signature.quantities == ()
    assert "nas1351-3-8" in signature.identifiers


def test_units_are_canonicalised_in_signatures() -> None:
    units = {
        text: semantic_signature(text).quantities[0].unit
        for text in (
            "Torque to 80 N·m.",
            "Torque to 80 lbf-in.",
            "Torque to 80 in-lb.",
            "Torque to 80 ft-lb.",
            "Limit 120 °C.",
            "Limit 3000 psi.",
            "Every 600 flight hours.",
        )
    }

    assert list(units.values()) == ["n_m", "in_lb", "in_lb", "ft_lb", "c", "psi", "hr"]


def test_month_may_is_not_a_modal_verb() -> None:
    assert not semantic_signature("Revision issued May 2024.").markers["modality"]
    assert not semantic_signature("Issued 5 May 2024.").markers["modality"]
    assert semantic_signature("The panel may be removed.").markers["modality"]


def test_figure_number_abbreviation_is_not_a_negation() -> None:
    figure = semantic_signature("Refer to Fig. No. 4 for details.")

    assert not figure.markers["negative"]
    assert semantic_signature("There is no leakage.").markers["negative"]
