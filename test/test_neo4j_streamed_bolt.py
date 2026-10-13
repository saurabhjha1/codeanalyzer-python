"""Streaming equivalence, incremental ownership, and bounded export buffers."""
import copy
import sys
import tracemalloc
from contextlib import nullcontext
from threading import Barrier, Lock
from types import SimpleNamespace

import pytest

from codeanalyzer.neo4j import bolt
from codeanalyzer.neo4j.project import ModuleSourceOwners, project
from codeanalyzer.neo4j.rows import NodeRef, RowBuilder, application_prefix
from codeanalyzer.schema.assign_ids import assign_ids
from sample_graph_app import make_sample_app


class MemoryDB:
    """Replay MERGE/SET and the writer's scoped purges, without a server."""

    def __init__(self):
        self.nodes, self.edges, self.queries = {}, {}, []
        self.max_nodes = self.max_edges = 0
        self.closed = False

    def session(self, **kwargs):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def execute_write(self, callback):
        return callback(self)

    def close(self):
        self.closed = True

    def consume(self):
        return self

    def single(self):
        return {"pruned": 0, "count": len(self.nodes)}

    def _delete(self, values):
        self.nodes = {k: n for k, n in self.nodes.items() if n.value not in values}
        self.edges = {k: e for k, e in self.edges.items()
                      if e.from_ref.value not in values and e.to_ref.value not in values}

    def run(self, query, **params):
        self.queries.append((query, copy.deepcopy(params)))
        if "RETURN m.id AS k" in query:
            return [{"k": n.value, "h": n.props.get("content_hash")}
                    for n in self.nodes.values() if "PyModule" in n.labels
                    and n.value.startswith(params["app"])]
        if query in (bolt.PURGE_MODULE_EDGES, bolt.PURGE_VANISHED_NODES):
            owned = {n.value for n in self.nodes.values()
                     if n.value == params["mid"] or n.value.startswith(params["pre"])}
            if query == bolt.PURGE_MODULE_EDGES:
                self.edges = {k: e for k, e in self.edges.items() if e.from_ref.value not in owned}
            else:
                self._delete(owned - set(params["keys"]))
        if query == bolt.PRUNE_VANISHED_MODULES:
            vanished = [n.value for n in self.nodes.values() if "PyModule" in n.labels
                        and n.value.startswith(params["app"]) and n.value not in params["present"]]
            self._delete({n.value for n in self.nodes.values()
                          if any(n.value == mid or n.value.startswith(mid + "/") for mid in vanished)})
        return self

    def upsert_nodes(self, session, neo4j, nodes):
        self.max_nodes = max(self.max_nodes, len(nodes))
        for n in nodes:
            key = (n.labels[0], n.value)
            if key in self.nodes:
                old = self.nodes[key]
                old.props.update(n.props)
                old.labels = sorted(set(old.labels) | set(n.labels))
            else:
                self.nodes[key] = copy.deepcopy(n)

    def upsert_edges(self, session, neo4j, edges):
        self.max_edges = max(self.max_edges, len(edges))
        for e in edges:
            # MATCH only creates a relationship if both nodes exist.
            if not all((ref.label, ref.value) in self.nodes for ref in (e.from_ref, e.to_ref)):
                continue
            key = (e.type, e.from_ref, e.to_ref, e.key)
            if key in self.edges:
                self.edges[key].props.update(e.props)
            else:
                self.edges[key] = copy.deepcopy(e)

    def graph(self):
        return (
            {k: (sorted(n.labels), n.props) for k, n in self.nodes.items()},
            {k: e.props for k, e in self.edges.items()},
        )


def bind(monkeypatch, db):
    monkeypatch.setitem(sys.modules, "neo4j", SimpleNamespace(
        GraphDatabase=SimpleNamespace(driver=lambda *args, **kwargs: db)))
    monkeypatch.setattr(bolt, "_upsert_nodes", db.upsert_nodes)
    monkeypatch.setattr(bolt, "_upsert_edges", db.upsert_edges)


def push(app, sig, eager=False, full=True):
    bolt.bolt_writer_streaming(app, "sample-app", sig, None,
                               bolt.BoltConfig("bolt://unused", "u", "p"), full, eager)


@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("full", [False, True])
def test_incremental_graph_matches_accumulating_writer(monkeypatch, eager, full):
    legacy, streamed = MemoryDB(), MemoryDB()
    app, sig = make_sample_app()
    # A second module exercises changed/unchanged ownership across module boundaries.
    app.symbol_table["other.py"] = copy.deepcopy(next(iter(app.symbol_table.values())))
    app.symbol_table["other.py"].content_hash = "other-hash"
    sig = assign_ids(app, "sample-app")
    versions = [copy.deepcopy(app), copy.deepcopy(app), copy.deepcopy(app), copy.deepcopy(app)]
    versions[2].symbol_table["service.py"].functions.pop("helper")
    versions[2].symbol_table["service.py"].content_hash = "changed-hash"
    del versions[3].symbol_table["other.py"]
    for version in versions:
        sig = assign_ids(version, "sample-app")
        bind(monkeypatch, legacy)
        bolt.bolt_writer(project(version, "sample-app", sig),
                         bolt.BoltConfig("bolt://unused", "u", "p"), full, eager)
        bind(monkeypatch, streamed)
        push(version, sig, eager, full)
        assert streamed.graph() == legacy.graph()
    assert streamed.closed


def test_streaming_buffers_and_reemissions(monkeypatch):
    db = MemoryDB()
    bind(monkeypatch, db)
    monkeypatch.setattr(bolt, "BATCH", 3)
    monkeypatch.setattr(bolt, "EDGE_BATCH", 3)
    builder = bolt._BoltNodeStreamBuilder(db.session, None, set(), application_prefix("a"), False)
    expected = RowBuilder()
    for i in range(100):
        labels = ["PySymbol"] + (["PyExternal"] if i % 2 else [])
        args = (labels, "id", f"can://a/@external/{i % 7}", {"version": i})
        builder.node(*args)
        expected.node(*args)
        assert len(builder._pending) < 3
        assert not builder._nodes and not builder._edges and not builder._deferred
    builder.finish()
    for row in expected.finish().nodes:
        actual = db.nodes[(row.labels[0], row.value)]
        assert actual.props == row.props
        assert set(actual.labels) == set(row.labels)
    assert db.max_nodes <= 3
    edges = bolt._BoltEdgeStreamBuilder(db.session, None, set(), builder._keys,
                                       builder.symbol_owners, SimpleNamespace(get=lambda _: None))
    source = NodeRef("PySymbol", "id", "can://a/@external/0")
    for i in range(100):
        target = NodeRef("PySymbol", "id", f"can://a/@external/{i % 7}")
        edges.edge("PY_DDG", source, target, {"n": i}, key=str(i % 2))
        edges.edge_to_symbol("PY_EXTENDS", source, NodeRef("PySymbol", "id", "missing"))
        assert len(edges._pending) < 3
        assert not edges._nodes and not edges._edges and not edges._deferred
    edges.finish()
    assert db.max_edges <= 3
    assert all(e.to_ref.value != "missing" for e in db.edges.values())
    assert {e.key for e in db.edges.values()} == {"0", "1"}


def test_module_boundaries_include_program_nodes_and_preserve_projection():
    app, sig = make_sample_app()
    class Boundaries(RowBuilder):
        current = None
        def begin_module(self, key, mod):
            self.current = key
        def end_module(self):
            self.current = None
        def node(self, labels, key, value, props):
            if props.get("_module") is not None:
                assert props["_module"] == self.current
            return super().node(labels, key, value, props)
    old = project(app, "sample-app", sig)
    new = project(app, "sample-app", sig, builder=Boundaries(), module_scoped=True)
    assert old.nodes == new.nodes
    assert old.edges == new.edges
    owners = ModuleSourceOwners(app)
    for n in old.nodes:
        if n.module:
            assert owners.get(n.value) == n.module
    mid = next(iter(app.symbol_table.values())).id
    assert owners.get(mid + "X/f()@1:0") is None


def test_program_graph_edges_carry_known_source_ownership():
    app, sig = make_sample_app()
    rows = project(app, "sample-app", sig)
    edges = [e for e in rows.edges if e.type in
             {"PY_HAS_BODY_NODE", "PY_CFG_NEXT", "PY_CDG", "PY_DDG", "PY_SUMMARY"}]
    assert edges
    assert all(e.from_ref.module in app.symbol_table for e in edges)


def test_fresh_eager_push_skips_purges_but_partial_retry_reconciles(monkeypatch):
    app, sig = make_sample_app()
    db = MemoryDB()
    bind(monkeypatch, db)
    push(app, sig, eager=True)
    purges = {bolt.PURGE_MODULE_EDGES, bolt.PURGE_VANISHED_NODES, bolt.PRUNE_VANISHED_MODULES}
    assert not any(q in purges for q, _ in db.queries)
    # Simulate a partial previous push: rows exist, but its completion hash
    # was never committed. A missing hash must not imply an empty database.
    for node in db.nodes.values():
        if "PyModule" in node.labels:
            node.props.pop("content_hash", None)
    db.queries.clear()
    push(app, sig, eager=True)
    assert purges <= {q for q, _ in db.queries}


def test_connection_failure_precedes_projection(monkeypatch):
    app, sig = make_sample_app()
    db = MemoryDB()
    bind(monkeypatch, db)
    def fail(*args, **kwargs):
        raise ConnectionError("bad Neo4j URI")
    db.run = fail
    import importlib
    projection = importlib.import_module("codeanalyzer.neo4j.project")
    monkeypatch.setattr(projection, "project", lambda *a, **k: pytest.fail("projected before connection"))
    with pytest.raises(ConnectionError):
        push(app, sig)
    assert db.closed


@pytest.mark.parametrize("workers", [1, 2])
def test_body_node_volume_does_not_grow_export_heap(monkeypatch, workers):
    # A sink that does not retain rows isolates export overhead from the input
    # analysis and database. Body IDs must not enter the persistent symbol keys.
    def discard_nodes(session, neo4j, rows):
        assert len(rows) <= bolt.BATCH
    def discard_edges(session, neo4j, rows):
        assert len(rows) <= bolt.EDGE_BATCH
    monkeypatch.setattr(bolt, "_upsert_nodes", discard_nodes)
    monkeypatch.setattr(bolt, "_upsert_edges", discard_edges)
    nodes = bolt._BoltNodeStreamBuilder(None, None, set(), application_prefix("a"), False)
    tracemalloc.start()
    try:
        for i in range(20000):
            nodes.node(["PyBodyNode"], "id", f"can://a/python/m.py/f()@{i}:0", {"kind": "statement"})
        nodes.finish()
        session = lambda: nullcontext(SimpleNamespace(execute_write=lambda fn: fn(None)))
        args = (session, None, set(), nodes._keys, nodes.symbol_owners,
                SimpleNamespace(get=lambda _: None))
        edges = (bolt._ParallelBoltEdgeStreamBuilder(*args, workers) if workers > 1
                 else bolt._BoltEdgeStreamBuilder(*args))
        try:
            for i in range(20000):
                edges.edge("PY_DDG", NodeRef("PyBodyNode", "id", f"src-{i}"),
                           NodeRef("PyBodyNode", "id", f"dst-{i}"), {"var": "x"}, key="x|ssa")
            edges.finish()
        finally:
            if workers > 1:
                edges.close()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert not nodes._keys and not nodes.symbol_owners
    assert peak < 32 * 1024 * 1024, f"export retained {peak} bytes for 40k rows"


def test_parallel_writes_overlap_and_preserve_repeated_relationship_order(monkeypatch):
    monkeypatch.setattr(bolt, "EDGE_BATCH", 2)
    sources = {}
    for i in range(100):
        ref = NodeRef("PyBodyNode", "id", str(i))
        sources.setdefault(hash(ref) % 2, ref)
    assert len(sources) == 2
    versions = {ref: [] for ref in sources.values()}
    barrier, lock = Barrier(2), Lock()
    def write(session, neo4j, rows):
        assert len(rows) <= 2
        if rows[0].props["version"] == 0:
            barrier.wait(timeout=5)  # both partitions must execute concurrently
        with lock:
            for row in rows:
                versions[row.from_ref].append(row.props["version"])
    monkeypatch.setattr(bolt, "_upsert_edges", write)
    db = MemoryDB()
    builder = bolt._ParallelBoltEdgeStreamBuilder(db.session, None, set(), set(), {},
                                                  SimpleNamespace(get=lambda _: None), 2)
    try:
        for version in range(8):
            for ref in sources.values():
                builder.edge("PY_DDG", ref, ref, {"version": version}, key="x|ssa")
        builder.finish()
    finally:
        builder.close()
    assert all(values == list(range(8)) for values in versions.values())
    assert builder.emitted == 16


def test_larger_relationship_batches_keep_nodes_at_their_smaller_bound(monkeypatch):
    node_sizes, edge_sizes = [], []
    monkeypatch.setattr(bolt, "_upsert_nodes", lambda s, n, rows: node_sizes.append(len(rows)))
    monkeypatch.setattr(bolt, "_upsert_edges", lambda s, n, rows: edge_sizes.append(len(rows)))
    nodes = bolt._BoltNodeStreamBuilder(None, None, set(), application_prefix("a"), False)
    edges = bolt._BoltEdgeStreamBuilder(None, None, set(), set(), {}, SimpleNamespace(get=lambda _: None))
    source = NodeRef("PyBodyNode", "id", "source")
    for i in range(2 * bolt.EDGE_BATCH + 7):
        nodes.node(["PyBodyNode"], "id", str(i), {"kind": "statement"})
        edges.edge("PY_DDG", source, NodeRef("PyBodyNode", "id", str(i)), key=str(i))
    nodes.finish()
    edges.finish()
    assert max(node_sizes) == bolt.BATCH
    assert max(edge_sizes) == bolt.EDGE_BATCH
    assert len(edge_sizes) < len(node_sizes)
    assert sum(node_sizes) == sum(edge_sizes) == 2 * bolt.EDGE_BATCH + 7


def test_failed_push_does_not_publish_hash_and_retry_completes(monkeypatch):
    app, sig = make_sample_app()
    db = MemoryDB()
    bind(monkeypatch, db)
    def fail(*args, **kwargs):
        raise ConnectionError("connection dropped during relationships")
    monkeypatch.setattr(bolt, "_upsert_edges", fail)
    with pytest.raises(ConnectionError):
        push(app, sig, eager=True)
    module = next(n for n in db.nodes.values() if "PyModule" in n.labels)
    assert "content_hash" not in module.props
    bind(monkeypatch, db)
    push(app, sig, eager=True)
    expected = MemoryDB()
    bind(monkeypatch, expected)
    bolt.bolt_writer(project(app, "sample-app", sig),
                     bolt.BoltConfig("bolt://unused", "u", "p"), True, True)
    assert db.graph() == expected.graph()
