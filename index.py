"""Chunking and hybrid retrieval: BM25 keywords + optional semantic embeddings, fused with Reciprocal Rank Fusion."""
from __future__ import annotations

import html as _html
import importlib.util
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

TOKEN = re.compile(r"\w+", re.UNICODE)
SENT = re.compile(r"(?<=[.!?۔؟])\s+")
STOP = set("""a an the and or but if of to in on at by for with from as is are was were be been being this that these
those it its i you he she we they them his her their our your my me not no do does did have has had will would can could
should may might about into over than then so such also there here which who whom what when where why how""".split())


def stem(t: str) -> str:
    """Light English stemmer: plants/plant, borrowing/borrowed/borrow, postgraduates/postgraduate all meet."""
    if not t.isascii() or len(t) < 4:
        return t
    if t.endswith("ies") and len(t) > 4:
        t = t[:-3] + "y"
    elif t.endswith(("sses", "xes", "zes", "ches", "shes")):
        t = t[:-2]
    elif t.endswith("ing") and len(t) > 5:
        t = t[:-3]
    elif t.endswith("ed") and len(t) > 4:
        t = t[:-2]
    elif t.endswith("s") and not t.endswith(("ss", "us", "is")):
        t = t[:-1]
    return t[:-1] if t.endswith("e") and len(t) > 4 else t


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in TOKEN.findall(text.lower()) if len(t) > 1 and t not in STOP]


def highlight(text: str, query: str) -> str:
    """HTML-escape text and wrap words that share a stem with a query word in <mark>."""
    out = _html.escape(text)
    stems = {s for s in tokenize(query) if len(s) > 2 and s not in {"amp", "quot"}}
    if not stems:
        return out
    pat = re.compile(r"(?<![&\w])(\w+)", re.UNICODE)
    return pat.sub(lambda m: f"<mark>{m.group(1)}</mark>" if stem(m.group(1).lower()) in stems else m.group(1), out)


class BM25:
    """Okapi BM25 with a always-positive IDF (works on tiny and large collections alike)."""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.n, self.k1, self.b = len(docs), k1, b
        self.dl = np.array([len(d) for d in docs], dtype=float)
        self.avg = float(self.dl.mean()) if self.n else 1.0
        self.post: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, d in enumerate(docs):
            for t, f in Counter(d).items():
                self.post[t].append((i, f))

    def scores(self, query: list[str]) -> np.ndarray:
        s = np.zeros(self.n)
        for t in set(query):
            plist = self.post.get(t)
            if not plist:
                continue
            idf = math.log(1 + (self.n - len(plist) + 0.5) / (len(plist) + 0.5))
            for i, f in plist:
                s[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.dl[i] / (self.avg or 1)))
        return s


# ───────────── chunking ─────────────
def _units(text: str, size: int) -> list[str]:
    units = []
    for line in re.split(r"\n+", text):
        line = line.strip()
        if not line:
            continue
        if len(line) <= size:
            units.append(line)
            continue
        for s in SENT.split(line):
            s = s.strip()
            while len(s) > size:
                units.append(s[:size])
                s = s[size:]
            if s:
                units.append(s)
    return units


def _is_heading(u: str) -> bool:
    return u.startswith("# ") and len(u) < 120


def chunk_text(text: str, size: int = 700, overlap: int = 120) -> list[str]:
    """Split into ~size-character chunks. Sections start new chunks and carry their heading for context."""
    chunks: list[str] = []
    cur: list[str] = []
    n, heading = 0, ""

    def flush():
        if any(not _is_heading(x) for x in cur):
            chunks.append("\n".join(cur))

    for u in _units(text, size):
        if _is_heading(u):
            if cur and n > 150:
                flush()
                cur, n = [], 0
            elif cur and all(_is_heading(x) for x in cur):
                cur, n = [], 0
            heading = u
        elif cur and n + len(u) + 1 > size:
            flush()
            keep, k = [], 0
            for x in reversed(cur):  # carry the last lines into the next chunk for context
                if _is_heading(x) or k + len(x) > overlap:
                    break
                keep.insert(0, x)
                k += len(x)
            cur = ([heading] if heading else []) + keep
            n = sum(len(x) + 1 for x in cur)
        cur.append(u)
        n += len(u) + 1
    flush()
    return chunks


@dataclass
class Chunk:
    id: int
    doc: str
    label: str
    text: str


@dataclass
class Hit:
    chunk: Chunk
    score: float      # 0-1, relative to the best match
    coverage: float   # share of query words found in the chunk


# ───────────── embeddings (optional) ─────────────
EMBED_MODELS = {"English (fast)": "BAAI/bge-small-en-v1.5",
                "Multilingual (incl. Urdu)": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"}


def embeddings_available() -> bool:
    try:
        return importlib.util.find_spec("fastembed") is not None
    except Exception:
        return False


class Embedder:
    """Small CPU embedding model (ONNX via fastembed). Downloads the model once on first use."""

    def __init__(self, model_name: str):
        from fastembed import TextEmbedding
        self.name = model_name
        self.m = TextEmbedding(model_name=model_name)

    @staticmethod
    def _norm(a) -> np.ndarray:
        a = np.array(list(a), dtype="float32")
        return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-9)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        fn = getattr(self.m, "passage_embed", self.m.embed)
        return self._norm(fn(texts))

    def embed_query(self, text: str) -> np.ndarray:
        fn = getattr(self.m, "query_embed", self.m.embed)
        return self._norm(fn([text]))[0]


# ───────────── the index ─────────────
class Index:
    def __init__(self):
        self.chunks: list[Chunk] = []
        self._tokens: list[list[str]] = []
        self._bm25 = None
        self.vec: dict[int, np.ndarray] = {}
        self.vec_model: str | None = None
        self._next = 0

    def add(self, name: str, pages: list, size: int = 700, overlap: int = 120) -> int:
        n = 0
        for label, text in pages:
            for c in chunk_text(text, size, overlap):
                self.chunks.append(Chunk(self._next, name, label, c))
                self._next += 1
                n += 1
        self._rebuild()
        return n

    def remove(self, name: str) -> None:
        gone = {c.id for c in self.chunks if c.doc == name}
        self.chunks = [c for c in self.chunks if c.doc != name]
        for i in gone:
            self.vec.pop(i, None)
        self._rebuild()

    def clear(self) -> None:
        self.__init__()

    def docs(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for c in self.chunks:
            out[c.doc] = out.get(c.doc, 0) + 1
        return out

    def _rebuild(self) -> None:
        self._tokens = [tokenize(c.text) for c in self.chunks]
        self._bm25 = BM25(self._tokens) if self.chunks else None

    def _ensure_vectors(self, emb: Embedder, progress=None) -> None:
        if self.vec_model != emb.name:
            self.vec, self.vec_model = {}, emb.name
        missing = [c for c in self.chunks if c.id not in self.vec]
        for i in range(0, len(missing), 64):
            batch = missing[i:i + 64]
            for c, v in zip(batch, emb.embed_passages([b.text for b in batch])):
                self.vec[c.id] = v
            if progress:
                progress(min(1.0, (i + 64) / len(missing)))

    def search(self, query: str, k: int = 6, embedder: Embedder | None = None, only: set | None = None,
               progress=None) -> list[Hit]:
        cand = [i for i, c in enumerate(self.chunks) if only is None or c.doc in only]
        if not cand or not query.strip():
            return []
        q = tokenize(query)
        bm = self._bm25.scores(q) if (self._bm25 and q) else np.zeros(len(self.chunks))
        order = [i for i in sorted(cand, key=lambda i: -bm[i]) if bm[i] > 0][:50]
        scores: dict[int, float] = {}
        for r, i in enumerate(order):
            scores[i] = scores.get(i, 0) + 1 / (60 + r)
        if embedder is not None:
            self._ensure_vectors(embedder, progress)
            qv = embedder.embed_query(query)
            sims = np.array([float(self.vec[self.chunks[i].id] @ qv) for i in cand])
            for r, j in enumerate(np.argsort(-sims)[:50]):
                scores[cand[j]] = scores.get(cand[j], 0) + 1 / (60 + r)
        if not scores:
            return []
        top = max(scores.values())
        qset, hits, seen = set(q), [], []
        for i in sorted(scores, key=lambda i: -scores[i]):
            toks = set(self._tokens[i])
            if any(len(toks & s) / max(1, len(toks | s)) > 0.85 for s in seen):  # skip near-duplicates
                continue
            seen.append(toks)
            hits.append(Hit(self.chunks[i], round(scores[i] / top, 3), round(len(qset & toks) / len(qset), 3) if qset else 0.0))
            if len(hits) == k:
                break
        return hits
