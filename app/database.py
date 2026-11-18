"""SQLite-backed revision storage and revision change stream.

Durability and concurrency contract:

* the database runs in WAL mode with ``synchronous=FULL``: every commit
  is fsync-ed and survives a process restart; readers never block on a
  writer and always observe a whole committed revision (or none of it);
* every write happens inside one ``BEGIN IMMEDIATE`` transaction which
  both serialises writers (across threads and processes) and makes the
  "revision must be strictly increasing" check atomic with the insert —
  a failed transaction rolls back completely, so storage errors can
  never leave a half revision;
* raw input and computed state are stored together, therefore a reader
  of any revision sees input, geometry and status from the same commit;
* every stored revision is paired with exactly one *change group* (the
  1°-grid cells whose water coverage changed) written in the **same**
  transaction, so the change stream can never miss a revision, hold a
  half group, or disagree with the serving state — including after
  rollbacks and restarts.  Group ids are assigned in commit order
  (writers are serialised) and rolled-back inserts consume no id, so
  committed groups form a gapless cursor sequence.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from typing import Callable, Iterator

SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS revisions (
        relation_id          INTEGER NOT NULL,
        revision             INTEGER NOT NULL,
        status               TEXT    NOT NULL CHECK (status IN ('valid', 'invalid')),
        input_json           TEXT    NOT NULL,
        geometry_json        TEXT,
        error_code           TEXT,
        error_message        TEXT,
        error_way_ids_json   TEXT,
        created_at           TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
        PRIMARY KEY (relation_id, revision)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS change_groups (
        group_id     INTEGER PRIMARY KEY AUTOINCREMENT,
        relation_id  INTEGER NOT NULL,
        revision     INTEGER NOT NULL,
        cells_json   TEXT    NOT NULL,
        created_at   TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
        UNIQUE (relation_id, revision),
        FOREIGN KEY (relation_id, revision)
            REFERENCES revisions (relation_id, revision)
    )
    """,
)

# No in-process lock is needed on purpose: SQLite's own BEGIN IMMEDIATE
# write lock serialises writers (across threads AND processes, e.g.
# multiple uvicorn workers), while WAL lets readers proceed concurrently
# without blocking a commit. A 30 s busy timeout makes contending writers
# queue instead of failing.
BUSY_TIMEOUT_SECONDS = 30.0
LOCK_RETRIES = 5


class Database:
    def __init__(self, path: str):
        self.path = path
        # check_same_thread=False: one short-lived connection per request,
        # shared nothing between threads.
        self._connect_kwargs = dict(
            timeout=BUSY_TIMEOUT_SECONDS,
            isolation_level=None,  # manual transaction control
            check_same_thread=False,
        )
        self.init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, **self._connect_kwargs)
        try:
            conn.row_factory = sqlite3.Row
            # Persistent WAL keeps working across restarted connections.
            # PRAGMAs must run outside an explicit transaction.
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=FULL")
            conn.execute("PRAGMA foreign_keys=ON")
            yield conn
        finally:
            conn.close()

    def init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                for statement in SCHEMA_STATEMENTS:
                    conn.execute(statement)
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    # ------------------------------------------------------------------ writes
    def save_revision(
        self,
        relation_id: int,
        revision: int,
        status: str,
        input_payload: dict,
        geometry: dict | None,
        error: dict | None,
        *,
        change_cells_fn: Callable[[dict | None], list[dict]] | None = None,
    ) -> int:
        """Atomically persist one full revision and its change group.

        ``change_cells_fn`` is called *inside* the write transaction with
        the previously latest stored revision row (or ``None``) and must
        return the 1°-grid cells whose coverage changed; computing it
        under the write lock guarantees the recorded diff matches the
        committed revision sequence even under concurrent uploads.  The
        change group commits or rolls back together with the revision.

        Returns the new change-group id.  Raises
        :class:`RevisionConflict` when ``revision`` is not strictly
        greater than every existing revision of the relation.
        """
        with self._connect() as conn:
            attempt = 0
            while True:
                try:
                    conn.execute("BEGIN IMMEDIATE")
                    break
                except sqlite3.OperationalError as exc:
                    # Another writer holds the lock; busy_timeout usually
                    # makes the execute block, but keep an explicit retry
                    # for the edge case where it returns immediately.
                    if "locked" in str(exc).lower() and attempt < LOCK_RETRIES:
                        attempt += 1
                        time.sleep(0.05 * attempt)
                        continue
                    raise
            try:
                prev_row = conn.execute(
                    "SELECT * FROM revisions WHERE relation_id = ? "
                    "ORDER BY revision DESC LIMIT 1",
                    (relation_id,),
                ).fetchone()
                if prev_row is not None and revision <= prev_row["revision"]:
                    conn.rollback()
                    raise RevisionConflict(
                        f"revision {revision} is not greater than the latest "
                        f"revision {prev_row['revision']}",
                        latest=prev_row["revision"],
                    )
                previous = self._row_to_dict(prev_row) if prev_row is not None else None
                cells = change_cells_fn(previous) if change_cells_fn is not None else []
                conn.execute(
                    """
                    INSERT INTO revisions (
                        relation_id, revision, status, input_json, geometry_json,
                        error_code, error_message, error_way_ids_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        relation_id,
                        revision,
                        status,
                        json.dumps(input_payload, separators=(",", ":"), ensure_ascii=False),
                        json.dumps(geometry, separators=(",", ":")) if geometry else None,
                        error["code"] if error else None,
                        error["message"] if error else None,
                        json.dumps(error["way_ids"]) if error else None,
                    ),
                )
                cursor = conn.execute(
                    """
                    INSERT INTO change_groups (relation_id, revision, cells_json)
                    VALUES (?, ?, ?)
                    """,
                    (
                        relation_id,
                        revision,
                        json.dumps(cells, separators=(",", ":")),
                    ),
                )
                group_id = cursor.lastrowid
                conn.commit()
                return group_id
            except RevisionConflict:
                raise
            except Exception:
                conn.rollback()
                raise

    # ------------------------------------------------------------------- reads
    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        error = None
        if row["error_code"]:
            error = {
                "code": row["error_code"],
                "message": row["error_message"],
                "way_ids": json.loads(row["error_way_ids_json"] or "[]"),
            }
        return {
            "relation_id": row["relation_id"],
            "revision": row["revision"],
            "status": row["status"],
            "input": json.loads(row["input_json"]),
            "geometry": json.loads(row["geometry_json"]) if row["geometry_json"] else None,
            "error": error,
            "created_at": row["created_at"],
        }

    def get_revision(self, relation_id: int, revision: int) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM revisions WHERE relation_id = ? AND revision = ?",
                (relation_id, revision),
            ).fetchone()
            return self._row_to_dict(row) if row else None

    def get_latest_revision(self, relation_id: int) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM revisions WHERE relation_id = ? "
                "ORDER BY revision DESC LIMIT 1",
                (relation_id,),
            ).fetchone()
            return self._row_to_dict(row) if row else None

    def get_last_valid(self, relation_id: int) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM revisions WHERE relation_id = ? AND status = 'valid' "
                "ORDER BY revision DESC LIMIT 1",
                (relation_id,),
            ).fetchone()
            return self._row_to_dict(row) if row else None

    def list_revisions(self, relation_id: int) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT revision, status FROM revisions WHERE relation_id = ? "
                "ORDER BY revision",
                (relation_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    # --------------------------------------------------------- change stream
    @staticmethod
    def _group_row_to_dict(row: sqlite3.Row) -> dict:
        return {
            "group_id": row["group_id"],
            "relation_id": row["relation_id"],
            "revision": row["revision"],
            "cells": json.loads(row["cells_json"]),
            "created_at": row["created_at"],
        }

    def get_change_groups(self, after: int = 0, limit: int = 100) -> list[dict]:
        """Committed change groups with ``group_id > after``, oldest first.

        Runs on a plain WAL read connection, so it never blocks writers
        and only ever observes fully committed groups.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM change_groups WHERE group_id > ? "
                "ORDER BY group_id LIMIT ?",
                (after, limit),
            ).fetchall()
            return [self._group_row_to_dict(r) for r in rows]

    def get_change_group_for_revision(
        self, relation_id: int, revision: int
    ) -> dict | None:
        """The single change group belonging to one stored revision."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM change_groups WHERE relation_id = ? AND revision = ?",
                (relation_id, revision),
            ).fetchone()
            return self._group_row_to_dict(row) if row else None


class RevisionConflict(Exception):
    def __init__(self, message: str, latest: int):
        super().__init__(message)
        self.latest = latest
