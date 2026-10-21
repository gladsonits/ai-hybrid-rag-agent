"""Turn text into vectors with a local Hugging Face sentence-transformer.

The model runs on the CPU by default, so indexing works without a paid API
key.  Model weights are pulled from the Hugging Face Hub on first use and
cached on disk; the free Hub download does not need Hub credits (only the
hosted inference API does).
"""

from __future__ import annotations

from collections.abc import Sequence

from retrieval.config import EmbeddingSettings


class EmbeddingModel:
    """A wrapper around a sentence-transformer that embeds batches of text.

    The wrapper owns three things the raw model does not: batching (so a
    158-chunk corpus is one forward pass per 32 chunks, not 158), a stable
    vector dimension, and L2 normalisation, which lets the pgvector index use
    the faster inner-product operator with cosine-equivalent results.
    """

    def __init__(self, settings: EmbeddingSettings | None = None) -> None:
        self.settings = settings or EmbeddingSettings.from_env()
        self._model = None
        self._dimension: int | None = None

    # -- lifecycle ---------------------------------------------------------

    def load(self) -> None:
        """Load the model into memory. Safe to call more than once."""
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(
            self.settings.model,
            device=self.settings.device,
            token=self.settings.hf_token or None,
        )
        self._model.max_seq_length = self.settings.max_seq_length
        self._dimension = int(self._model.get_sentence_embedding_dimension())

    @property
    def dimension(self) -> int:
        """Length of the vectors this model produces."""
        if self._dimension is None:
            self.load()
        assert self._dimension is not None
        return self._dimension

    # -- embedding ---------------------------------------------------------

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a list of strings, returning one vector per input.

        An empty input returns an empty list, so callers do not have to
        special-case it.
        """
        if not texts:
            return []
        self.load()
        vectors = self._model.encode(
            list(texts),
            batch_size=self.settings.batch_size,
            normalize_embeddings=self.settings.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [vector.tolist() for vector in vectors]

    def embed_one(self, text: str) -> list[float]:
        """Embed a single string."""
        return self.embed([text])[0]
