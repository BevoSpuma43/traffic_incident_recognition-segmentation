import csv
import json
import sqlite3
from dataclasses import asdict
from pathlib import Path


class EventStorage:
    def __init__(self, output_dir):
        self.root = Path(output_dir)
        self.events_dir = self.root / "events"
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.events_dir / "events.sqlite")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS events (event_id TEXT PRIMARY KEY, camera_id TEXT, "
            "impact_time_s REAL, run_id TEXT, payload TEXT NOT NULL)"
        )
        self.connection.commit()

    def save(self, event):
        payload = asdict(event)
        text = json.dumps(payload, indent=2, allow_nan=False)
        target = self.events_dir / f"{event.event_id}.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(target)
        self.connection.execute(
            "INSERT OR REPLACE INTO events VALUES (?, ?, ?, ?, ?)",
            (event.event_id, event.camera_id, event.impact_time_s, event.run_id, text),
        )
        self.connection.commit()

    def list_events(self, run_id=None, limit=500):
        query, args = "SELECT payload FROM events", []
        if run_id is not None:
            query += " WHERE run_id = ?"
            args.append(run_id)
        query += " ORDER BY impact_time_s DESC LIMIT ?"
        args.append(limit)
        return [json.loads(row[0]) for row in self.connection.execute(query, args)]

    def export(self, path, run_id=None):
        rows = self.list_events(run_id, limit=1_000_000)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".csv":
            with path.open("w", newline="", encoding="utf-8") as handle:
                if rows:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
        else:
            path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    def close(self):
        self.connection.close()
