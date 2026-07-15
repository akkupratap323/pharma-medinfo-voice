"""
Semantic Template Selector — intent understanding via the OpenAI embeddings API.

Replaces the local sentence-transformers / MiniLM model (which pulled ~800MB of
PyTorch into RAM at startup) with OpenAI `text-embedding-3-small` — the SAME
model already used for LightRAG. Template embeddings are computed once (one
batched call) and cached; each query costs one small embedding call
(~$0.00001, ~50-80ms). Cosine similarity is done in NumPy. No torch, no
sentence-transformers — the whole ML stack drops off the runtime.

Public API is unchanged: SemanticTemplateSelector, get_semantic_selector(),
is_semantic_available(), .select_template(), .select_template_with_fallback().
"""

import os
from typing import Dict, List, Optional, Tuple

import httpx
import numpy as np
from loguru import logger

_OPENAI_EMBED_URL = "https://api.openai.com/v1/embeddings"
_EMBED_MODEL = os.getenv("A2UI_EMBED_MODEL", "text-embedding-3-small")


def _openai_api_key() -> str:
    return os.getenv("OPENAI_API_KEY", "")


def _embed(texts: List[str]) -> List[np.ndarray]:
    """Embed a batch of texts via OpenAI. Returns L2-normalized float32 vectors
    (so cosine similarity is just a dot product)."""
    key = _openai_api_key()
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set — semantic selector requires it")
    with httpx.Client(timeout=20) as client:
        r = client.post(
            _OPENAI_EMBED_URL,
            headers={"Authorization": f"Bearer {key}"},
            json={"model": _EMBED_MODEL, "input": texts},
        )
        r.raise_for_status()
        data = r.json()["data"]
    vecs: List[np.ndarray] = []
    for item in sorted(data, key=lambda d: d["index"]):
        v = np.asarray(item["embedding"], dtype=np.float32)
        n = float(np.linalg.norm(v))
        vecs.append(v / n if n > 0 else v)
    return vecs


class SemanticTemplateSelector:
    """Selects the best A2UI template for a query via OpenAI embedding similarity.

    Attributes:
        template_embeddings: Pre-computed unit embeddings per template type
    """

    def __init__(self, model_name: str = _EMBED_MODEL):
        self.model_name = model_name
        self.template_embeddings: Dict[str, np.ndarray] = {}
        self.template_descriptions: Dict[str, str] = {}
        self._build_template_embeddings()
        logger.info(
            f"SemanticTemplateSelector ready (OpenAI {model_name}) — "
            f"{len(self.template_embeddings)} templates indexed"
        )

    def _build_template_embeddings(self):
        """Embed each template's metadata (name + description + use_cases +
        trigger_keywords) in ONE batched OpenAI call, cached for the session."""
        from .template_library import list_available_templates

        logger.debug("Building template embeddings from template_library metadata...")
        template_metadata = list_available_templates()

        types: List[str] = []
        descriptions: List[str] = []
        for template in template_metadata["templates"]:
            ttype = template["type"]
            parts = [template["name"], template["description"]]
            if template.get("use_cases"):
                parts.extend(template["use_cases"])
            if template.get("trigger_keywords"):
                parts.extend(template["trigger_keywords"])
            combined = " | ".join(parts)
            self.template_descriptions[ttype] = combined
            types.append(ttype)
            descriptions.append(combined)

        embeddings = _embed(descriptions)  # one batched request for all templates
        for ttype, emb in zip(types, embeddings):
            self.template_embeddings[ttype] = emb

        logger.info(f"✅ Pre-computed embeddings for {len(self.template_embeddings)} templates")

    def _scores(self, query: str) -> Dict[str, float]:
        """Cosine similarity (dot product of unit vectors) of query vs every template."""
        q = _embed([query])[0]
        return {
            ttype: float(np.dot(q, emb))
            for ttype, emb in self.template_embeddings.items()
        }

    def select_template(
        self,
        query: str,
        threshold: float = 0.3,
    ) -> Tuple[Optional[str], float]:
        """Return (template_type, confidence) for the best match, or (None, score)
        if nothing clears `threshold`.

        NOTE: text-embedding-3-small produces a different similarity distribution
        than MiniLM did; 0.3 is a reasonable starting threshold and the A2UI
        orchestrator falls back to keyword matching below it.
        """
        scores = self._scores(query)
        best_template, best_score = None, -1.0
        for ttype, score in scores.items():
            if score > best_score:
                best_score, best_template = score, ttype

        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.debug(f"🎯 Semantic scores for '{query[:60]}': " +
                     ", ".join(f"{t}={s:.3f}" for t, s in top))

        if best_score < threshold:
            logger.debug(f"⚠️ Best score ({best_score:.3f}) below threshold ({threshold})")
            return None, best_score

        logger.info(f"✅ Semantic match: {best_template} (confidence: {best_score:.3f})")
        return best_template, best_score

    def select_template_with_fallback(
        self,
        query: str,
        keyword_result: Optional[str] = None,
        semantic_threshold: float = 0.3,
        confidence_threshold: float = 0.5,
    ) -> Tuple[str, float, str]:
        """Semantic match with keyword fallback (unchanged policy)."""
        semantic_template, semantic_score = self.select_template(query, semantic_threshold)

        if semantic_score >= confidence_threshold:
            return semantic_template, semantic_score, "semantic"

        if semantic_template and semantic_threshold <= semantic_score < confidence_threshold:
            if keyword_result:
                return keyword_result, semantic_score, "keyword_fallback"
            return semantic_template, semantic_score, "semantic"

        if keyword_result:
            return keyword_result, 0.0, "keyword_fallback"
        return "magazine-hero", 0.0, "default"

    def get_all_scores(self, query: str) -> Dict[str, float]:
        """All template similarity scores, sorted desc (debugging)."""
        return dict(sorted(self._scores(query).items(), key=lambda x: x[1], reverse=True))


# Global instance (lazy loaded)
_semantic_selector: Optional[SemanticTemplateSelector] = None
_initialization_error: Optional[str] = None


def get_semantic_selector() -> Optional[SemanticTemplateSelector]:
    """Get or create the global selector. Builds template embeddings on first use
    (one OpenAI call). Returns None if OPENAI_API_KEY is missing or the call fails."""
    global _semantic_selector, _initialization_error

    if _initialization_error:
        logger.debug(f"Semantic selector unavailable: {_initialization_error}")
        return None

    if _semantic_selector is None:
        try:
            _semantic_selector = SemanticTemplateSelector()
        except Exception as e:  # noqa: BLE001 - degrade to keyword tier
            _initialization_error = f"Failed to initialize: {e}"
            logger.warning(f"⚠️ Semantic selector init failed (falling back to keywords): {e}")
            return None

    return _semantic_selector


def is_semantic_available() -> bool:
    """Available iff an OpenAI key is configured — a cheap check that does NOT
    import any heavy ML libs (this call runs at orchestrator import time)."""
    return bool(_openai_api_key())
