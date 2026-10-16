"""Batch routing over a CSV file, and accuracy scoring against labelled examples."""

import csv
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from notif_router.router import NotificationRouter
from notif_router.schemas import IncomingMessage, RoutingDecision

logger = logging.getLogger(__name__)

OUTPUT_COLUMNS = ["message_id", "action", "message_type", "reason", "confidence", "evidence_message_ids"]


def read_messages(path: Path) -> list[IncomingMessage]:
    with path.open(encoding="utf-8", newline="") as handle:
        fields = IncomingMessage.model_fields
        return [IncomingMessage(**{k: v for k, v in row.items() if k in fields})
                for row in csv.DictReader(handle)]


def route_file(
    router: NotificationRouter,
    input_path: Path,
    output_path: Path,
    use_cache: bool = True,
    progress: bool = True,
) -> list[RoutingDecision]:
    messages = read_messages(input_path)
    decisions = []
    for index, message in enumerate(messages, start=1):
        _, decision = router.route(message, use_cache=use_cache)
        decisions.append(decision)
        if progress:
            print(f"[{index:>3}/{len(messages)}] {message.message_id:<16} "
                  f"{decision.action.value:<6} {decision.message_type.value:<15} "
                  f"({decision.source.value}, {decision.confidence:.2f})")
    write_decisions(decisions, output_path)
    return decisions


def write_decisions(decisions: list[RoutingDecision], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(OUTPUT_COLUMNS)
        for d in decisions:
            writer.writerow([d.message_id, d.action.value, d.message_type.value, d.reason,
                             f"{d.confidence:.2f}", d.evidence_as_csv()])


@dataclass
class EvaluationReport:
    total: int
    action_accuracy: float
    type_accuracy: float
    both_correct: float
    evidence_overlap: float
    confusion: dict[str, Counter] = field(default_factory=dict)
    mistakes: list[dict] = field(default_factory=list)

    def render(self) -> str:
        lines = [
            f"Labelled messages : {self.total}",
            f"Action accuracy   : {self.action_accuracy:.1%}",
            f"Type accuracy     : {self.type_accuracy:.1%}",
            f"Both correct      : {self.both_correct:.1%}",
            f"Evidence overlap  : {self.evidence_overlap:.1%}  (labelled evidence we also cited)",
            "",
            "Action confusion (expected -> predicted):",
        ]
        for expected, predicted in sorted(self.confusion.items()):
            lines.append(f"  {expected:<7} -> " + ", ".join(f"{k}: {v}" for k, v in predicted.most_common()))
        if self.mistakes:
            lines += ["", "Mistakes:"]
            for m in self.mistakes:
                lines.append(f"  {m['message_id']:<16} expected {m['expected']:<26} got {m['predicted']}")
        return "\n".join(lines)


def evaluate(router: NotificationRouter, labelled_path: Path, use_cache: bool = True) -> EvaluationReport:
    with labelled_path.open(encoding="utf-8", newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if (r.get("action") or "").strip()]
    fields = IncomingMessage.model_fields

    action_hits = type_hits = both_hits = 0
    evidence_expected = evidence_found = 0
    confusion: dict[str, Counter] = {}
    mistakes = []
    for row in rows:
        message = IncomingMessage(**{k: v for k, v in row.items() if k in fields})
        _, decision = router.route(message, use_cache=use_cache)
        action_ok = decision.action.value == row["action"]
        type_ok = decision.message_type.value == row["message_type"]
        action_hits += action_ok
        type_hits += type_ok
        both_hits += action_ok and type_ok
        confusion.setdefault(row["action"], Counter())[decision.action.value] += 1

        expected_evidence = {e for e in row.get("evidence_message_ids", "").split(";") if e and e != "none"}
        evidence_expected += len(expected_evidence)
        evidence_found += len(expected_evidence & set(decision.evidence_message_ids))
        if not (action_ok and type_ok):
            mistakes.append({
                "message_id": message.message_id,
                "expected": f"{row['action']}/{row['message_type']}",
                "predicted": f"{decision.action.value}/{decision.message_type.value} ({decision.source.value})",
            })

    total = max(len(rows), 1)
    return EvaluationReport(
        total=len(rows),
        action_accuracy=action_hits / total,
        type_accuracy=type_hits / total,
        both_correct=both_hits / total,
        evidence_overlap=evidence_found / evidence_expected if evidence_expected else 0.0,
        confusion=confusion,
        mistakes=mistakes,
    )
