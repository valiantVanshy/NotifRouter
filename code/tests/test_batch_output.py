"""End-to-end check on the real dataset: the offline router must produce a valid output file."""

import csv

from notif_router.config import DEFAULT_DATASET_DIR
from notif_router.evaluation import OUTPUT_COLUMNS, read_messages, route_file
from notif_router.router import NotificationRouter


def test_offline_batch_covers_every_message(real_store, repository, tmp_path):
    router = NotificationRouter(real_store, repository, llm=None)
    output = tmp_path / "output.csv"
    route_file(router, DEFAULT_DATASET_DIR / "messages.csv", output, progress=False)

    expected_ids = {m.message_id for m in read_messages(DEFAULT_DATASET_DIR / "messages.csv")}
    with output.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == OUTPUT_COLUMNS
        rows = list(reader)

    assert {r["message_id"] for r in rows} == expected_ids
    assert len(rows) == len(expected_ids)
    for row in rows:
        assert row["action"] in {"notify", "digest", "mute"}
        assert 0.0 <= float(row["confidence"]) <= 1.0
        assert row["reason"].strip()
        evidence = row["evidence_message_ids"]
        assert evidence == "none" or all(e in real_store.history_by_id for e in evidence.split(";"))
