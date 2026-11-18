"""Tests for the 1°-grid revision change stream (GET /changes).

Covers: coverage semantics (positive area only, holes excluded, edge
contact excluded), the serving-state machine (takedown on invalid, no
double takedown, resume), unchanged-face uploads, cursor pagination,
atomicity with the revision commit, durability across restarts and
behaviour under concurrency.
"""

from __future__ import annotations

import sqlite3
import threading

from fastapi.testclient import TestClient

from app.database import Database
from app.main import app

from .factories import (
    INNER_NODES,
    INNER_WAY,
    SQUARE_NODES,
    SQUARE_WAY,
    node,
    revision,
    way,
)


def post(client: TestClient, relation_id: int, payload: dict):
    return client.post(f"/relations/{relation_id}/revisions", json=payload)


def get_changes(client: TestClient, after: int = 0, limit: int = 100) -> dict:
    resp = client.get("/changes", params={"after": after, "limit": limit})
    assert resp.status_code == 200
    return resp.json()


def all_groups(client: TestClient) -> list[dict]:
    return get_changes(client)["groups"]


def groups_for(client: TestClient, relation_id: int) -> list[dict]:
    return [g for g in all_groups(client) if g["relation_id"] == relation_id]


def cells_of(group: dict) -> set[tuple[int, int]]:
    return {(c["lon"], c["lat"]) for c in group["cells"]}


# Shifted copy of the unit square, one degree east.
MOVED_NODES = [node(1, 1.0, 0.0), node(2, 2.0, 0.0), node(3, 2.0, 1.0), node(4, 1.0, 1.0)]
MOVED_WAY = way(10, [1, 2, 3, 4, 1])

# Unit square shrunk to its south-west quarter (sub-cell change).
SHRUNK_NODES = [node(1, 0.0, 0.0), node(2, 0.5, 0.0), node(3, 0.5, 0.5), node(4, 0.0, 0.5)]
SHRUNK_WAY = way(10, [1, 2, 3, 4, 1])


# ------------------------------------------------------------- coverage semantics
def test_first_valid_upload_records_covered_cells(client: TestClient, relation_id: int):
    post(client, relation_id, revision(1, SQUARE_NODES, [SQUARE_WAY]))
    groups = groups_for(client, relation_id)
    assert len(groups) == 1
    assert groups[0]["revision"] == 1
    # The unit square [0,1]x[0,1] covers exactly cell (0, 0); the cells
    # starting at lon=1 / lat=1 only share an edge and are excluded.
    assert cells_of(groups[0]) == {(0, 0)}


def test_edge_only_contact_is_not_coverage(client: TestClient, relation_id: int):
    # Polygon exactly [1,2]x[0,1]: touches cell (0,0) along the lon=1 edge.
    resp = post(client, relation_id, revision(1, MOVED_NODES, [MOVED_WAY]))
    assert resp.json()["status"] == "valid"
    groups = groups_for(client, relation_id)
    assert cells_of(groups[0]) == {(1, 0)}


def test_inner_ring_is_not_coverage(client: TestClient, relation_id: int):
    # Hole fills the whole cell (0,0)-overlapping area except a frame;
    # a hole alone never produces coverage.  Use a polygon whose hole
    # spans an entire cell: outer [0,2]x[0,2], inner [0,1]x[0,1]... the
    # inner ring must stay strictly inside, so shrink it slightly.
    outer_nodes = [
        node(1, 0.0, 0.0),
        node(2, 2.0, 0.0),
        node(3, 2.0, 2.0),
        node(4, 0.0, 2.0),
    ]
    hole_nodes = [
        node(5, 0.001, 0.001),
        node(6, 0.999, 0.001),
        node(7, 0.999, 0.999),
        node(8, 0.001, 0.999),
    ]
    ways = [way(10, [1, 2, 3, 4, 1]), way(20, [5, 6, 7, 8, 5], role="inner")]
    post(client, relation_id, revision(1, outer_nodes + hole_nodes, ways))
    groups = groups_for(client, relation_id)
    # Cells (0,0),(1,0),(0,1),(1,1) all keep positive coverage (the hole
    # never covers a full cell), so all four are listed on first upload.
    assert cells_of(groups[0]) == {(0, 0), (1, 0), (0, 1), (1, 1)}


def test_hole_change_marks_hole_cells(client: TestClient, relation_id: int):
    post(client, relation_id, revision(1, SQUARE_NODES, [SQUARE_WAY]))
    # Revision 2 adds a hole: coverage inside the hole disappears.
    post(client, relation_id, revision(2, SQUARE_NODES + INNER_NODES, [SQUARE_WAY, INNER_WAY]))
    groups = groups_for(client, relation_id)
    assert len(groups) == 2
    # The hole [0.2,0.8]^2 lies entirely in cell (0,0): that cell's
    # coverage changed (positive area lost) -> listed.
    assert cells_of(groups[1]) == {(0, 0)}


def test_subcell_area_change_marks_the_cell(client: TestClient, relation_id: int):
    post(client, relation_id, revision(1, SQUARE_NODES, [SQUARE_WAY]))
    # Revision 2 shrinks the polygon to its SW quarter: cell (0,0) stays
    # covered but loses 0.75 deg^2 of water -> positive-area change.
    post(client, relation_id, revision(2, SHRUNK_NODES, [SHRUNK_WAY]))
    groups = groups_for(client, relation_id)
    assert cells_of(groups[1]) == {(0, 0)}


def test_moved_polygon_marks_symmetric_difference(client: TestClient, relation_id: int):
    post(client, relation_id, revision(1, SQUARE_NODES, [SQUARE_WAY]))
    post(client, relation_id, revision(2, MOVED_NODES, [MOVED_WAY]))
    groups = groups_for(client, relation_id)
    # Cell (0,0) lost its water, cell (1,0) gained water.
    assert cells_of(groups[1]) == {(0, 0), (1, 0)}


def test_negative_coordinates_map_to_negative_cells(client: TestClient, relation_id: int):
    nodes = [
        node(1, -1.5, -1.5),
        node(2, -0.5, -1.5),
        node(3, -0.5, -0.5),
        node(4, -1.5, -0.5),
    ]
    post(client, relation_id, revision(1, nodes, [way(10, [1, 2, 3, 4, 1])]))
    groups = groups_for(client, relation_id)
    # Spans lon [-1.5,-0.5] -> cells -2 and -1; lat likewise.
    assert cells_of(groups[0]) == {(-2, -2), (-1, -2), (-2, -1), (-1, -1)}


def test_polygon_spanning_many_cells(client: TestClient, relation_id: int):
    nodes = [
        node(1, -1.0, -1.0),
        node(2, 2.0, -1.0),
        node(3, 2.0, 2.0),
        node(4, -1.0, 2.0),
    ]
    post(client, relation_id, revision(1, nodes, [way(10, [1, 2, 3, 4, 1])]))
    groups = groups_for(client, relation_id)
    expected = {(lon, lat) for lon in (-1, 0, 1) for lat in (-1, 0, 1)}
    assert cells_of(groups[0]) == expected


# --------------------------------------------------------- unchanged face uploads
def test_reordered_ways_same_face_produce_no_cells(client: TestClient, relation_id: int):
    post(client, relation_id, revision(1, SQUARE_NODES, [SQUARE_WAY]))
    # Same square, split into 4 unordered ways, one reversed.
    ways = [
        way(11, [2, 3]),
        way(12, [1, 4]),
        way(13, [4, 3]),
        way(14, [1, 2]),
    ]
    resp = post(client, relation_id, revision(2, SQUARE_NODES, ways))
    assert resp.json()["status"] == "valid"
    groups = groups_for(client, relation_id)
    assert len(groups) == 2  # every upload still forms a group
    assert groups[1]["cells"] == []


def test_identical_reupload_produces_empty_group(client: TestClient, relation_id: int):
    post(client, relation_id, revision(1, SQUARE_NODES, [SQUARE_WAY]))
    post(client, relation_id, revision(2, SQUARE_NODES, [SQUARE_WAY]))
    groups = groups_for(client, relation_id)
    assert groups[1]["cells"] == []


# ------------------------------------------------------------- state machine diffs
def test_invalid_revision_takes_down_old_face(client: TestClient, relation_id: int):
    post(client, relation_id, revision(1, SQUARE_NODES, [SQUARE_WAY]))
    post(client, relation_id, revision(2, SQUARE_NODES, [way(10, [1, 2, 3, 99, 1])]))
    assert client.get(f"/relations/{relation_id}/current").status_code == 409
    groups = groups_for(client, relation_id)
    assert len(groups) == 2
    assert groups[1]["revision"] == 2
    # The old face's cells become uncovered.
    assert cells_of(groups[1]) == {(0, 0)}


def test_consecutive_invalid_revisions_do_not_repeat_takedown(
    client: TestClient, relation_id: int
):
    post(client, relation_id, revision(1, SQUARE_NODES, [SQUARE_WAY]))
    post(client, relation_id, revision(2, SQUARE_NODES, [way(10, [1, 2, 3, 99, 1])]))
    post(client, relation_id, revision(3, SQUARE_NODES, [way(10, [1, 2, 3, 98, 1])]))
    groups = groups_for(client, relation_id)
    assert len(groups) == 3
    assert cells_of(groups[1]) == {(0, 0)}  # takedown once ...
    assert groups[2]["cells"] == []  # ... not twice


def test_first_revision_invalid_records_no_cells(client: TestClient, relation_id: int):
    post(client, relation_id, revision(1, SQUARE_NODES, [way(10, [1, 2, 3, 99, 1])]))
    groups = groups_for(client, relation_id)
    assert len(groups) == 1
    assert groups[0]["cells"] == []


def test_valid_revision_after_invalid_records_restored_cells(
    client: TestClient, relation_id: int
):
    post(client, relation_id, revision(1, SQUARE_NODES, [SQUARE_WAY]))
    post(client, relation_id, revision(2, SQUARE_NODES, [way(10, [1, 2, 3, 99, 1])]))
    post(client, relation_id, revision(3, MOVED_NODES, [MOVED_WAY]))
    groups = groups_for(client, relation_id)
    assert len(groups) == 3
    # Serving resumed with the moved face: only its cells appear (the old
    # face was already taken down by revision 2).
    assert cells_of(groups[2]) == {(1, 0)}


def test_every_accepted_revision_has_exactly_one_group(client: TestClient, relation_id: int):
    post(client, relation_id, revision(1, SQUARE_NODES, [SQUARE_WAY]))
    post(client, relation_id, revision(2, SQUARE_NODES, [way(10, [1, 2, 3, 99, 1])]))
    post(client, relation_id, revision(3, SQUARE_NODES + INNER_NODES, [SQUARE_WAY, INNER_WAY]))

    listing = client.get(f"/relations/{relation_id}/revisions").json()["revisions"]
    groups = groups_for(client, relation_id)
    assert [g["revision"] for g in groups] == [r["revision"] for r in listing]
    # Rejected uploads (conflict) produce no group.
    conflict = post(client, relation_id, revision(3, SQUARE_NODES, [SQUARE_WAY]))
    assert conflict.status_code == 409
    assert len(groups_for(client, relation_id)) == 3


# ------------------------------------------------------------------ cursor reads
def test_cursor_pagination_walks_the_whole_stream(client: TestClient, relation_id: int):
    other_relation = relation_id + 1
    for rev in range(1, 6):
        post(client, relation_id, revision(rev, SQUARE_NODES, [SQUARE_WAY]))
    post(client, other_relation, revision(1, MOVED_NODES, [MOVED_WAY]))

    seen: list[int] = []
    cursor = 0
    while True:
        page = get_changes(client, after=cursor, limit=2)
        if not page["groups"]:
            assert page["next_cursor"] == cursor
            break
        for group in page["groups"]:
            assert group["group_id"] > cursor
            seen.append(group["group_id"])
        cursor = page["next_cursor"]
    assert len(seen) == 6
    assert seen == sorted(seen)
    assert len(set(seen)) == 6  # no group delivered twice

    # Group ids are gapless: committed in order, rollbacks consume none.
    assert seen == list(range(seen[0], seen[0] + 6))

    # A cursor in the middle of the stream resumes exactly there.
    page = get_changes(client, after=seen[2])
    assert [g["group_id"] for g in page["groups"]] == seen[3:]


def test_changes_query_params_validated(client: TestClient):
    assert client.get("/changes", params={"after": -1}).status_code == 422
    assert client.get("/changes", params={"limit": 0}).status_code == 422
    assert client.get("/changes", params={"limit": 1001}).status_code == 422


def test_empty_stream_returns_cursor_unchanged(client: TestClient):
    page = get_changes(client, after=0)
    assert page == {"groups": [], "next_cursor": 0}


# --------------------------------------------------------------------- atomicity
def test_failed_revision_insert_leaves_no_change_group(db_path: str, relation_id: int):
    db = Database(db_path)
    good = revision(1, SQUARE_NODES, [SQUARE_WAY])
    db.save_revision(relation_id, 1, "valid", good, {"type": "Polygon"}, None)

    try:
        db.save_revision(relation_id, 2, "bogus-status", good, None, None)  # type: ignore[arg-type]
    except Exception:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected storage failure")

    assert db.get_revision(relation_id, 2) is None
    assert db.get_change_group_for_revision(relation_id, 2) is None
    groups = db.get_change_groups()
    assert len(groups) == 1
    assert groups[0]["revision"] == 1


def test_revision_conflict_consumes_no_group_id(db_path: str, relation_id: int):
    db = Database(db_path)
    good = revision(1, SQUARE_NODES, [SQUARE_WAY])
    db.save_revision(relation_id, 1, "valid", good, {"type": "Polygon"}, None)
    try:
        db.save_revision(relation_id, 1, "valid", good, {"type": "Polygon"}, None)
    except Exception:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected conflict")
    db.save_revision(relation_id, 2, "valid", good, {"type": "Polygon"}, None)
    ids = [g["group_id"] for g in db.get_change_groups()]
    assert ids == list(range(ids[0], ids[0] + 2))  # still gapless


def test_change_group_commits_atomically_with_revision(db_path: str, relation_id: int):
    db = Database(db_path)
    payload = revision(1, SQUARE_NODES, [SQUARE_WAY])
    geometry = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}

    def boom(_previous):
        raise RuntimeError("diff computation failed")

    try:
        db.save_revision(
            relation_id, 1, "valid", payload, geometry, None, change_cells_fn=boom
        )
    except RuntimeError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected failure")
    # The revision insert rolled back together with the group.
    assert db.get_revision(relation_id, 1) is None
    assert db.get_change_groups() == []


# --------------------------------------------------------------------- durability
def test_change_groups_survive_restart(client: TestClient, db_path: str, relation_id: int):
    post(client, relation_id, revision(1, SQUARE_NODES, [SQUARE_WAY]))
    post(client, relation_id, revision(2, SQUARE_NODES, [way(10, [1, 2, 3, 99, 1])]))

    restarted = Database(db_path)
    app.state.db = restarted

    groups = restarted.get_change_groups()
    assert len(groups) == 2
    assert cells_of(groups[0]) == {(0, 0)}
    assert cells_of(groups[1]) == {(0, 0)}  # takedown of the old face
    # The stream stays consistent with the serving state after restart.
    assert restarted.get_latest_revision(relation_id)["status"] == "invalid"


def test_change_stream_consistent_with_serving_state_after_restart(
    client: TestClient, db_path: str, relation_id: int
):
    post(client, relation_id, revision(1, SQUARE_NODES, [SQUARE_WAY]))
    post(client, relation_id, revision(2, MOVED_NODES, [MOVED_WAY]))
    post(client, relation_id, revision(3, SQUARE_NODES, [way(10, [1, 2, 3, 99, 1])]))

    app.state.db = Database(db_path)
    resp = client.get("/changes")
    groups = [g for g in resp.json()["groups"] if g["relation_id"] == relation_id]
    assert [cells_of(g) for g in groups] == [{(0, 0)}, {(0, 0), (1, 0)}, {(1, 0)}]
    assert client.get(f"/relations/{relation_id}/current").status_code == 409


# --------------------------------------------------------------------- concurrency
def test_concurrent_uploads_produce_complete_consistent_groups(
    client: TestClient, relation_id: int
):
    revisions = list(range(1, 17))
    lock = threading.Lock()
    accepted: list[int] = []

    def upload(rev: int) -> None:
        # Alternate the square and the moved square so most accepted
        # revisions change coverage.
        payload = revision(
            rev,
            SQUARE_NODES if rev % 2 else MOVED_NODES,
            [SQUARE_WAY],
        )
        r = post(client, relation_id, payload)
        assert r.status_code in (200, 409), r.text
        if r.status_code == 200:
            with lock:
                accepted.append(rev)

    threads = [threading.Thread(target=upload, args=(rev,)) for rev in revisions]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    stored = client.get(f"/relations/{relation_id}/revisions").json()["revisions"]
    stored_revs = [r["revision"] for r in stored]
    assert sorted(accepted) == stored_revs

    groups = groups_for(client, relation_id)
    # Exactly one group per committed revision, in commit order, gapless.
    assert len(groups) == len(stored_revs)
    assert {g["revision"] for g in groups} == set(stored_revs)
    ids = [g["group_id"] for g in groups]
    assert ids == sorted(ids)
    assert ids == list(range(ids[0], ids[0] + len(ids)))

    # Each group's cells match the transition between the revision
    # committed immediately before it and its own revision.
    by_revision = {g["revision"]: g for g in groups}
    for idx, rev in enumerate(stored_revs):
        group = by_revision[rev]
        prev_rev = stored_revs[idx - 1] if idx else None
        expected = _expected_cells(client, relation_id, prev_rev, rev)
        assert cells_of(group) == expected, f"revision {rev}"


def _expected_cells(
    client: TestClient, relation_id: int, prev_rev: int | None, rev: int
) -> set[tuple[int, int]]:
    """Recompute the expected diff from the stored revisions themselves."""
    from app.changes import changed_cells, serving_geometry

    prev_row = (
        client.get(f"/relations/{relation_id}/revisions/{prev_rev}").json()
        if prev_rev is not None
        else None
    )
    row = client.get(f"/relations/{relation_id}/revisions/{rev}").json()
    old = serving_geometry(prev_row)
    new = serving_geometry(row)
    return {(c["lon"], c["lat"]) for c in changed_cells(old, new)}


def test_concurrent_reads_never_see_partial_groups(client: TestClient, db, relation_id: int):
    post(client, relation_id, revision(1, SQUARE_NODES, [SQUARE_WAY]))
    stop = threading.Event()
    anomalies: list[str] = []

    reader_client = TestClient(app)
    reader_client.__enter__()
    app.state.db = db

    def reader() -> None:
        while not stop.is_set():
            page = reader_client.get("/changes").json()
            for group in page["groups"]:
                if not isinstance(group["cells"], list):  # pragma: no cover
                    anomalies.append("cells not a list")
                if group["revision"] < 1:  # pragma: no cover
                    anomalies.append("bogus revision")

    t = threading.Thread(target=reader)
    t.start()
    try:
        for rev in range(2, 12):
            r = post(client, relation_id, revision(rev, SQUARE_NODES, [SQUARE_WAY]))
            assert r.status_code == 200
    finally:
        stop.set()
        t.join(timeout=10)
        reader_client.__exit__(None, None, None)

    assert not t.is_alive()
    assert anomalies == []
    groups = groups_for(client, relation_id)
    assert len(groups) == 11
    # Re-uploads of the identical face: only the first group has cells.
    assert cells_of(groups[0]) == {(0, 0)}
    assert all(g["cells"] == [] for g in groups[1:])


# ------------------------------------------------------------------ docs / schema
def test_changes_endpoint_in_openapi_docs(client: TestClient):
    assert client.get("/docs").status_code == 200
    schema = client.get("/openapi.json").json()
    assert "/changes" in schema["paths"]
    get_op = schema["paths"]["/changes"]["get"]
    param_names = {p["name"] for p in get_op["parameters"]}
    assert param_names == {"after", "limit"}
    assert "ChangeStream" in schema["components"]["schemas"]
    assert "ChangeGroup" in schema["components"]["schemas"]
    assert "ChangeCell" in schema["components"]["schemas"]


def test_existing_endpoints_still_work(client: TestClient, relation_id: int):
    # The original face endpoints are unaffected by the change stream.
    post(client, relation_id, revision(1, SQUARE_NODES, [SQUARE_WAY]))
    current = client.get(f"/relations/{relation_id}/current")
    assert current.status_code == 200
    assert current.json()["geometry"]["type"] == "Polygon"
    rev = client.get(f"/relations/{relation_id}/revisions/1")
    assert rev.status_code == 200
    assert rev.json()["input"]["revision"] == 1
    listing = client.get(f"/relations/{relation_id}/revisions")
    assert listing.json()["revisions"] == [{"revision": 1, "status": "valid"}]
    assert client.get("/health").status_code == 200


def test_wal_mode_still_active_with_change_groups(db_path: str):
    Database(db_path)
    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        assert {"revisions", "change_groups"} <= tables
    finally:
        conn.close()
