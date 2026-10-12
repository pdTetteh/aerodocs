"""Fixed, deterministic research questions for a defect report."""

from dataclasses import dataclass

from app.work_orders.models import DefectReport, FindingPurpose

MAX_SUBJECT_CHARACTERS = 300


@dataclass(frozen=True)
class PlannedQuestion:
    finding_id: str
    purpose: FindingPurpose
    question: str


def plan_questions(defect: DefectReport) -> list[PlannedQuestion]:
    """Build the fixed question set for a defect.

    The question wording is a server-owned template. Only the short subject
    comes from the report, and no step is chosen by the model, so the research
    plan cannot be steered by defect text or retrieved documents.
    """
    subject = defect.title
    if defect.component and defect.component.casefold() not in subject.casefold():
        subject = f"{subject} ({defect.component})"
    subject = subject[:MAX_SUBJECT_CHARACTERS].strip()
    templates: tuple[tuple[FindingPurpose, str], ...] = (
        (
            "inspection",
            f"What inspection does the documentation specify for: {subject}?",
        ),
        (
            "limits",
            f"What limits or tolerances does the documentation give for: {subject}?",
        ),
        (
            "repair_or_replacement",
            "What repair or replacement procedure does the documentation "
            f"describe for: {subject}?",
        ),
        (
            "required_approvals",
            f"What approval does the documentation require for: {subject}?",
        ),
    )
    return [
        PlannedQuestion(finding_id=f"finding_{index}", purpose=purpose, question=text)
        for index, (purpose, text) in enumerate(templates, start=1)
    ]
