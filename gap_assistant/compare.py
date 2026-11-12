"""Step 2: compare papers - method x dataset matrix, untried combinations, recurring
limitations / future work, and a 2-D "topic map"."""
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA

from .embeddings import Embedder
from .schemas import Paper

PLACEHOLDERS = {"", "na", "none", "notspecified", "notapplicable", "unknown", "notstated",
                "notreported", "nan", "null"}

# cosine-distance thresholds for "these two statements say the same thing"
CLUSTER_DISTANCE = {"tfidf": 0.8, "gemini": 0.4}


def _key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _pid_order(pid: str) -> int:
    m = re.search(r"\d+", pid)
    return int(m.group()) if m else 0


def collect_items(papers: List[Paper], field_name: str) -> Dict[str, dict]:
    """Group the values of a list field (methods, datasets, ...) across papers, ignoring
    case/punctuation differences. -> {key: {"name": display name, "papers": [ids]}}"""
    names: Dict[str, Counter] = defaultdict(Counter)
    ids: Dict[str, set] = defaultdict(set)
    for p in papers:
        for item in getattr(p, field_name):
            k = _key(item)
            if k in PLACEHOLDERS:
                continue
            names[k][item.strip()] += 1
            ids[k].add(p.paper_id)
    return {
        k: {"name": names[k].most_common(1)[0][0], "papers": sorted(ids[k], key=_pid_order)}
        for k in names
    }


def cooccurrence_matrix(papers: List[Paper], rows: str = "methods", cols: str = "datasets",
                        top_n: int = 15) -> pd.DataFrame:
    """Number of papers that use each (row item, column item) pair. Paper-level, so a paper
    listing several methods and datasets counts every pair - treat as an overview, not proof."""
    r, c = collect_items(papers, rows), collect_items(papers, cols)
    top_r = sorted(r, key=lambda k: (-len(r[k]["papers"]), r[k]["name"].lower()))[:top_n]
    top_c = sorted(c, key=lambda k: (-len(c[k]["papers"]), c[k]["name"].lower()))[:top_n]
    df = pd.DataFrame(0, index=[r[k]["name"] for k in top_r], columns=[c[k]["name"] for k in top_c])
    for rk in top_r:
        for ck in top_c:
            df.loc[r[rk]["name"], c[ck]["name"]] = len(set(r[rk]["papers"]) & set(c[ck]["papers"]))
    return df


def coverage_hints(papers: List[Paper], min_freq: int = 2, max_hints: int = 12) -> List[dict]:
    """Method/dataset pairs that are each common in the set but never used together."""
    m, d = collect_items(papers, "methods"), collect_items(papers, "datasets")
    together = set()
    for p in papers:
        mk = {_key(x) for x in p.methods} - PLACEHOLDERS
        dk = {_key(x) for x in p.datasets} - PLACEHOLDERS
        together |= {(a, b) for a in mk for b in dk}
    hints = []
    for mk, mv in m.items():
        if len(mv["papers"]) < min_freq:
            continue
        for dk, dv in d.items():
            if len(dv["papers"]) < min_freq or (mk, dk) in together:
                continue
            hints.append({
                "method": mv["name"], "dataset": dv["name"],
                "method_papers": mv["papers"], "dataset_papers": dv["papers"],
                "weight": len(mv["papers"]) * len(dv["papers"]),
            })
    hints.sort(key=lambda h: (-h["weight"], h["method"].lower(), h["dataset"].lower()))
    return hints[:max_hints]


def recurring_statements(papers: List[Paper], kind: str, embedder: Embedder) -> List[dict]:
    """Cluster limitation / future-work statements and keep themes raised by >= 2 papers."""
    items = []
    for p in papers:
        for c in getattr(p, kind):
            items.append({"paper_id": p.paper_id, "statement": c.statement,
                          "quote": c.quote, "verified": c.verified})
    if len({i["paper_id"] for i in items}) < 2:
        return []
    vecs = embedder.embed([i["statement"] for i in items])
    threshold = CLUSTER_DISTANCE[embedder.last_backend]
    labels = AgglomerativeClustering(
        n_clusters=None, metric="cosine", linkage="average", distance_threshold=threshold
    ).fit_predict(vecs)

    themes = []
    for label in sorted(set(labels)):
        idx = [i for i, lab in enumerate(labels) if lab == label]
        members = [items[i] for i in idx]
        pids = sorted({m["paper_id"] for m in members}, key=_pid_order)
        if len(pids) < 2:
            continue
        centroid = vecs[idx].mean(axis=0)
        rep = idx[int(np.argmax(vecs[idx] @ centroid))]
        themes.append({"theme": items[rep]["statement"], "paper_ids": pids,
                       "n_papers": len(pids), "items": members})
    themes.sort(key=lambda t: (-t["n_papers"], t["theme"]))
    return themes


def paper_map(papers: List[Paper], embedder: Embedder) -> pd.DataFrame:
    """2-D positions for a scatter plot: similar papers sit close together."""
    cols = ["paper_id", "title", "x", "y", "cluster"]
    if len(papers) < 2:
        return pd.DataFrame(columns=cols)
    texts = [f"{p.domain}. {p.research_problem} Methods: {', '.join(p.methods)}. "
             f"Datasets: {', '.join(p.datasets)}" for p in papers]
    vecs = embedder.embed(texts)
    n_comp = min(2, vecs.shape[0], vecs.shape[1])
    coords = PCA(n_components=n_comp, random_state=0).fit_transform(vecs)
    if coords.shape[1] < 2:
        coords = np.hstack([coords, np.zeros((coords.shape[0], 1))])
    k = min(max(2, len(papers) // 4), 6) if len(papers) >= 4 else 1
    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(vecs) if k > 1 else [0] * len(papers)
    return pd.DataFrame({
        "paper_id": [p.paper_id for p in papers],
        "title": [p.title for p in papers],
        "x": coords[:, 0], "y": coords[:, 1],
        "cluster": [f"Group {int(label) + 1}" for label in labels],
    })


@dataclass
class ComparisonResult:
    matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    coverage_hints: List[dict] = field(default_factory=list)
    recurring_limitations: List[dict] = field(default_factory=list)
    recurring_future_work: List[dict] = field(default_factory=list)
    paper_map: pd.DataFrame = field(default_factory=pd.DataFrame)
    method_counts: Dict[str, int] = field(default_factory=dict)
    dataset_counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "matrix": {"index": list(self.matrix.index), "columns": list(self.matrix.columns),
                       "values": self.matrix.values.tolist()},
            "coverage_hints": self.coverage_hints,
            "recurring_limitations": self.recurring_limitations,
            "recurring_future_work": self.recurring_future_work,
            "method_counts": self.method_counts,
            "dataset_counts": self.dataset_counts,
        }


def build_comparison(papers: List[Paper], embedder: Embedder) -> ComparisonResult:
    methods, datasets = collect_items(papers, "methods"), collect_items(papers, "datasets")
    return ComparisonResult(
        matrix=cooccurrence_matrix(papers),
        coverage_hints=coverage_hints(papers),
        recurring_limitations=recurring_statements(papers, "limitations", embedder),
        recurring_future_work=recurring_statements(papers, "future_work", embedder),
        paper_map=paper_map(papers, embedder),
        method_counts={v["name"]: len(v["papers"]) for v in methods.values()},
        dataset_counts={v["name"]: len(v["papers"]) for v in datasets.values()},
    )
