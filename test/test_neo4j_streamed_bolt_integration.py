"""Opt-in equivalence against an isolated Neo4j database (clears that database).

Set CANPY_TEST_NEO4J_URI and CANPY_TEST_NEO4J_PASSWORD for a disposable server.
"""
import copy
import json
import os

import pytest

from codeanalyzer.neo4j.bolt import BoltConfig, bolt_writer, bolt_writer_streaming
from codeanalyzer.neo4j.project import project
from codeanalyzer.schema.assign_ids import assign_ids
from sample_graph_app import make_sample_app

pytestmark = pytest.mark.skipif(not os.getenv("CANPY_TEST_NEO4J_URI"),
                                reason="requires an explicitly supplied disposable Neo4j")


def snapshot(driver):
    with driver.session() as s:
        nodes = [{"labels": sorted(r["labels"]), "props": r["props"]}
                 for r in s.run("MATCH (n) RETURN labels(n) AS labels, properties(n) AS props")]
        edges = [{"from": r["a"], "from_labels": sorted(r["al"]), "type": r["t"],
                  "props": r["p"], "to": r["b"], "to_labels": sorted(r["bl"])}
                 for r in s.run("MATCH (a)-[r]->(b) RETURN properties(a) AS a, labels(a) AS al, "
                                "type(r) AS t, properties(r) AS p, properties(b) AS b, labels(b) AS bl")]
    return sorted(json.dumps(n, sort_keys=True) for n in nodes), sorted(json.dumps(e, sort_keys=True) for e in edges)


@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("full", [False, True])
def test_live_incremental_equivalence(eager, full):
    neo4j = pytest.importorskip("neo4j")
    cfg = BoltConfig(os.environ["CANPY_TEST_NEO4J_URI"], "neo4j",
                     os.environ["CANPY_TEST_NEO4J_PASSWORD"])
    driver = neo4j.GraphDatabase.driver(cfg.uri, auth=(cfg.user, cfg.password))
    app, _ = make_sample_app()
    app.symbol_table["other.py"] = copy.deepcopy(app.symbol_table["service.py"])
    app.symbol_table["other.py"].content_hash = "other-hash"
    original = copy.deepcopy(app)
    changed = copy.deepcopy(app)
    changed.symbol_table["service.py"].functions.pop("helper")
    changed.symbol_table["service.py"].content_hash = "changed-hash"
    removed = copy.deepcopy(changed)
    del removed.symbol_table["other.py"]
    try:
        results = []
        for streaming in (False, True):
            with driver.session() as s:
                s.run("MATCH (n) DETACH DELETE n").consume()
            # A sibling application with colliding source paths must survive
            # both per-module reconciliation and full-run orphan pruning.
            sibling = copy.deepcopy(original)
            sibling_sig = assign_ids(sibling, "other-app")
            bolt_writer(project(sibling, "other-app", sibling_sig), cfg, True, True)
            snapshots = []
            for version in (original, original, changed, removed):
                version = copy.deepcopy(version)
                sig = assign_ids(version, "sample-app")
                if streaming:
                    bolt_writer_streaming(version, "sample-app", sig, None, cfg, full, eager)
                else:
                    bolt_writer(project(version, "sample-app", sig), cfg, full, eager)
                snapshots.append(snapshot(driver))
            results.append(snapshots)
        assert results[0] == results[1]
    finally:
        driver.close()
