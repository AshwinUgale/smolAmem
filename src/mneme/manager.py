"""The :class:`MemoryManager` facade.

Wires a backend, an embedder, and the three tiers together for one agent.
Exposes the tiers as attributes — ``manager.working``, ``manager.episodic``,
``manager.semantic`` — so callers can reach for the exact tier they need,
plus a single :meth:`retrieve` that fuses results from the persisted tiers
with authority weighting + freshness decay.

Construction is the one place Mneme assembles all its moving parts:

.. code-block:: python

    from mneme import (
        MemoryManager, OpenAIEmbeddings, SQLiteBackend,
    )

    embedder = OpenAIEmbeddings()
    backend = SQLiteBackend(path="memories.db", dimensions=embedder.dimensions)
    manager = MemoryManager(
        agent_id="alice", backend=backend, embedder=embedder,
    )

    manager.working.add(role="user", content="hi there")
    episode = manager.episodic.add("user asked about React Suspense")
    fact = manager.semantic.add("user is working on a Next.js project")

    # Single fused retrieval across episodic + semantic.
    hits = manager.retrieve("React Suspense boundaries", k=5)

Working memory is **not part of** :meth:`retrieve`. Access it directly via
``manager.working.turns()`` and append the turns to your prompt.
"""

from __future__ import annotations

import contextlib
import math
from datetime import UTC, datetime, timedelta
from typing import Any

from mneme.backends import MnemeBackend
from mneme.embeddings import EmbeddingProvider
from mneme.judge import LLMJudge
from mneme.tiers import EpisodicMemoryTier, SemanticMemoryTier, WorkingMemoryTier
from mneme.types import EpisodicMemory, MemoryTier, RetrievalResult, SemanticFact

# Default per-tier authority. Semantic facts outrank episodic recall because
# a consolidated fact is a higher-confidence signal than a single past
# utterance. Working is here for completeness but never used — retrieve()
# refuses to search working memory.
DEFAULT_AUTHORITY_WEIGHTS: dict[MemoryTier, float] = {
    MemoryTier.SEMANTIC: 1.0,
    MemoryTier.EPISODIC: 0.7,
    MemoryTier.WORKING: 0.0,
}

# Default freshness half-life. A 7-day-old record is worth half its
# similarity score; a 14-day-old record is worth a quarter; and so on.
# Override per-call when an agent's memories age faster or slower.
DEFAULT_HALF_LIFE_DAYS = 7.0

# Multiplier applied to ``k`` when over-fetching from the backend so the
# final top-k after re-ranking is meaningful.
_OVERFETCH_FACTOR = 4

# ---------------------------------------------------------------------------
# Consolidation defaults (v0.2)
# ---------------------------------------------------------------------------

# How many recent episodes the consolidator hands the LLM judge per call.
# Smaller batches → more LLM calls, finer-grained extraction; larger batches
# → fewer calls, lower cost, slightly less attention per episode.
DEFAULT_CONSOLIDATION_BATCH_SIZE = 20

# Cosine-similarity threshold for treating a newly extracted fact as a
# duplicate of an existing semantic fact. Above the threshold → merge
# (newer-wins-with-provenance); below → store as a new fact.
DEFAULT_DEDUP_THRESHOLD = 0.85

# Hard cap on how many episodes consolidation will read in a single run.
# Prevents runaway costs on very long agent histories. Override per call.
DEFAULT_CONSOLIDATION_MAX_EPISODES = 200

# ---------------------------------------------------------------------------
# Forgetting defaults (v0.3)
# ---------------------------------------------------------------------------

# A record is "cold" if it hasn't been accessed in this many days AND its
# access_count is at-or-below the floor. Defaults are conservative: a record
# never retrieved in 30 days qualifies; one that's been retrieved at least
# once survives by default.
DEFAULT_COLD_AGE_DAYS = 30
DEFAULT_ACCESS_FLOOR = 0

# Hard cap on how many records the forgetting pass scans per tier per run.
# Forgetting is O(n) over records (we have to look at each to decide); past
# this size, run forget() more often or raise the cap explicitly.
DEFAULT_FORGET_MAX_PER_TIER = 5000

# Tiers the forgetting pass walks. Working memory lives in process and is
# evicted by FIFO; the persisted tiers are what get forgotten on disk.
_FORGETTABLE_TIERS = (MemoryTier.EPISODIC, MemoryTier.SEMANTIC)


# The structured-output schema for fact extraction. Lives at module scope so
# tests can introspect it. ``additionalProperties: false`` + explicit
# ``required`` lists keep OpenAI's strict mode happy.
_FACT_EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The fact in one natural-language sentence.",
                    },
                    "confidence": {
                        "type": "number",
                        "description": "How sure consolidation is, in [0.0, 1.0].",
                    },
                    "source_episode_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Ids of the episodes that justify this fact.",
                    },
                },
                "required": ["content", "confidence", "source_episode_ids"],
            },
        },
    },
    "required": ["facts"],
}

_CONSOLIDATION_SYSTEM_PROMPT = """\
You are a memory consolidator for an LLM agent. Your job is to read a batch
of recent conversation episodes and extract durable facts about the user,
their project, or their preferences — things worth remembering for weeks or
months, not transient details.

Episodes are tagged with timestamps and IDs.

Extract facts that are:
- Specific (not "user is interested in tech" but "user is building a Next.js \
14 app with the app router")
- Likely to remain true (not "user is tired" but "user works in San Francisco")
- Useful for future context (something a future agent answer should know)

For each fact, output:
- content: the fact in one natural-language sentence
- confidence: 0.0 to 1.0 based on how clearly the episodes support it
- source_episode_ids: the IDs of the episodes that contributed to this fact

Output an empty list if no durable facts can be extracted from this batch.
Do not fabricate facts the episodes do not support.
"""


class MemoryManager:
    """Owns one agent's three tiers and the shared backend + embedder.

    Args:
        agent_id: The namespace for everything this manager touches. Passed
            through to each tier so backend writes are isolated.
        backend: Storage + vector-search for episodic + semantic tiers. Must
            be configured for ``embedder.dimensions``.
        embedder: Embedding provider used for both write-time (add) and
            query-time (search) vectors.
        working_size: Max turns kept in working memory. Default 20.
        llm_judge: Optional :class:`LLMJudge` used by :meth:`consolidate`.
            ``None`` (default) is fine if you never call ``consolidate()``;
            it raises ``RuntimeError`` when invoked without a judge.
    """

    def __init__(
        self,
        *,
        agent_id: str,
        backend: MnemeBackend,
        embedder: EmbeddingProvider,
        working_size: int = 20,
        llm_judge: LLMJudge | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.backend = backend
        self.embedder = embedder
        self.llm_judge = llm_judge

        self.working = WorkingMemoryTier(agent_id=agent_id, max_size=working_size)
        self.episodic = EpisodicMemoryTier(agent_id=agent_id, backend=backend, embedder=embedder)
        self.semantic = SemanticMemoryTier(agent_id=agent_id, backend=backend, embedder=embedder)

        # Lazy: set on first start_scheduler() call (None means "no scheduler").
        # Typed as Any so the apscheduler import stays lazy.
        self._scheduler: Any = None

    # ---------------------------------------------------------------------
    # Retrieval
    # ---------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        *,
        k: int = 5,
        tiers: list[MemoryTier] | None = None,
        authority_weights: dict[MemoryTier, float] | None = None,
        half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
        use_confidence: bool = True,
        now: datetime | None = None,
    ) -> list[RetrievalResult]:
        """Return the top-``k`` memories most relevant to ``query``.

        Embeds the query once, asks the backend for an over-fetched candidate
        set across the requested tiers, then re-ranks each candidate by

        .. math::

            score = similarity \\times authority \\times recency \\times confidence?

        where:

        * ``similarity`` is the raw cosine similarity returned by the backend
          (clamped to a non-negative floor so negative-similarity records
          never beat positive-similarity ones on tier or recency boosts).
        * ``authority`` is :data:`DEFAULT_AUTHORITY_WEIGHTS` for the record's
          tier, or the override in ``authority_weights``.
        * ``recency`` is ``0.5 ** (age_days / half_life_days)`` — exponential
          decay with a configurable half-life.
        * ``confidence?`` is the :attr:`SemanticFact.confidence` value if the
          record is a fact and ``use_confidence`` is ``True``; otherwise 1.

        Args:
            query: The natural-language query to embed and match against.
            k: Maximum number of results. Default 5.
            tiers: Which tiers to search. Default ``[EPISODIC, SEMANTIC]``.
                Passing ``MemoryTier.WORKING`` raises ``ValueError`` —
                working memory has its own access path
                (``manager.working.turns()``) and is not embedded.
            authority_weights: Override per-tier authority. Missing tiers
                default to 0.0. Use to express domain-specific policy
                (e.g., raise EPISODIC weight for an agent where the
                conversation history matters more than distilled facts).
            half_life_days: Days after which a record's recency factor drops
                to 0.5. Default 7. Smaller = faster forgetting.
            use_confidence: Whether to multiply semantic facts' scores by
                their :attr:`SemanticFact.confidence`. Default True. Disable
                to make confidence a metadata-only signal.
            now: Reference time for recency computation. Defaults to
                ``datetime.now(UTC)``. Pass explicitly in tests.

        Returns:
            A list of :class:`RetrievalResult`, ordered by ``score``
            descending. Each result carries the raw ``similarity``,
            computed ``recency``, and ``authority`` so callers can inspect
            why a record ranked where it did.
        """
        if k <= 0:
            return []

        selected_tiers = (
            list(tiers) if tiers is not None else [MemoryTier.EPISODIC, MemoryTier.SEMANTIC]
        )
        if MemoryTier.WORKING in selected_tiers:
            raise ValueError(
                "Working memory is not searchable through retrieve(). "
                "Use manager.working.turns() to read the recent turns directly."
            )
        if not selected_tiers:
            return []

        weights = dict(authority_weights) if authority_weights else dict(DEFAULT_AUTHORITY_WEIGHTS)
        reference_time = now if now is not None else datetime.now(UTC)

        # One embedding call (vs two if we routed through the tier wrappers).
        [query_embedding] = self.embedder.embed([query])

        # One backend call covering all selected tiers. Over-fetch so re-ranking
        # has headroom; the top-k by raw similarity is not always the top-k
        # after authority + recency apply.
        raw = self.backend.search(
            query_embedding=query_embedding,
            agent_id=self.agent_id,
            k=k * _OVERFETCH_FACTOR,
            tiers=selected_tiers,
        )

        results: list[RetrievalResult] = []
        for record, similarity in raw:
            # Defensive expiration filter: even if forget() hasn't run yet,
            # retrieve() should never surface a record past its TTL.
            if record.expires_at is not None and record.expires_at <= reference_time:
                continue
            # Clamp at zero so dissimilar records never get boosted by tier or
            # recency into a misleadingly positive final score.
            effective_similarity = max(0.0, similarity)
            authority = weights.get(record.tier, 0.0)
            recency = _recency_decay(
                created_at=record.created_at,
                now=reference_time,
                half_life_days=half_life_days,
            )
            confidence_mul = 1.0
            if use_confidence and isinstance(record, SemanticFact):
                confidence_mul = record.confidence

            score = effective_similarity * authority * recency * confidence_mul
            results.append(
                RetrievalResult(
                    record=record,
                    score=score,
                    similarity=similarity,
                    recency=recency,
                    authority=authority,
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        top = results[:k]

        # Persist access tracking for the records we actually returned.
        # ``backend.touch`` is authoritative — that's what the forgetting
        # pass reads. For backends where the in-memory record IS the
        # storage object (e.g. InMemoryBackend), the caller-held record
        # reflects the new state automatically via shared reference. For
        # detached-copy backends (SQLite), the in-memory record's
        # ``access_count`` is one behind until the next read; callers who
        # need the post-retrieve count should ``backend.get`` it.
        if top:
            self.backend.touch([r.record.id for r in top], now=reference_time)

        return top

    # ---------------------------------------------------------------------
    # Consolidation (v0.2)
    # ---------------------------------------------------------------------

    def consolidate(
        self,
        *,
        since: datetime | None = None,
        batch_size: int = DEFAULT_CONSOLIDATION_BATCH_SIZE,
        dedup_threshold: float = DEFAULT_DEDUP_THRESHOLD,
        max_episodes: int = DEFAULT_CONSOLIDATION_MAX_EPISODES,
    ) -> list[SemanticFact]:
        """Promote recent episodic memories into semantic facts.

        Reads up to ``max_episodes`` episodes (filtered by ``since`` when
        given), batches them, calls :attr:`llm_judge` per batch with the
        fact-extraction prompt, then for each extracted fact:

        1. Embeds the fact content.
        2. Searches existing semantic facts for the most similar one.
        3. If best-similarity ≥ ``dedup_threshold`` → merge into the existing
           fact (newer content wins; provenance is the union; confidence is
           the max).
        4. Otherwise → write the fact as a new :class:`SemanticFact`.

        Args:
            since: If given, only episodes with ``created_at >= since`` are
                considered. ``None`` reads from the very first episode.
            batch_size: Episodes per LLM call. Default 20.
            dedup_threshold: Cosine similarity above which two facts are
                considered duplicates. Default 0.85.
            max_episodes: Hard cap on how many episodes a single
                ``consolidate()`` run will process. Default 200.

        Returns:
            The :class:`SemanticFact` records that were either created or
            merged-into during this run.

        Raises:
            RuntimeError: if ``self.llm_judge is None``.
        """
        if self.llm_judge is None:
            raise RuntimeError(
                "MemoryManager has no llm_judge — consolidation is unavailable. "
                "Pass llm_judge= to MemoryManager(...) to enable it."
            )
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if max_episodes <= 0:
            return []

        episodes = self.backend.list_recent(
            agent_id=self.agent_id,
            tier=MemoryTier.EPISODIC,
            since=since,
            limit=max_episodes,
        )
        if not episodes:
            return []

        # Process oldest → newest within the window so the consolidator sees
        # cause-and-effect in the order it happened.
        episodes_sorted = sorted(episodes, key=lambda e: e.created_at)
        episodes_typed: list[EpisodicMemory] = [
            e for e in episodes_sorted if isinstance(e, EpisodicMemory)
        ]

        affected: list[SemanticFact] = []
        for start in range(0, len(episodes_typed), batch_size):
            batch = episodes_typed[start : start + batch_size]
            extracted = _extract_facts_from_batch(judge=self.llm_judge, episodes=batch)
            for candidate in extracted:
                fact = _store_or_merge_fact(
                    semantic=self.semantic,
                    embedder=self.embedder,
                    candidate=candidate,
                    dedup_threshold=dedup_threshold,
                )
                affected.append(fact)
        return affected

    # ---------------------------------------------------------------------
    # Forgetting (v0.3)
    # ---------------------------------------------------------------------

    def forget(
        self,
        *,
        now: datetime | None = None,
        ttl_only: bool = False,
        access_floor: int = DEFAULT_ACCESS_FLOOR,
        cold_age_days: float = DEFAULT_COLD_AGE_DAYS,
        max_per_tier: int = DEFAULT_FORGET_MAX_PER_TIER,
    ) -> dict[str, int]:
        """Delete expired and cold records from the persisted tiers.

        Two phases, in order:

        1. **TTL:** every record whose ``expires_at`` is set and ``<= now``
           is deleted unconditionally. This is the only thing that runs
           when ``ttl_only=True``.
        2. **Access-frequency decay:** records older than ``cold_age_days``
           with ``access_count <= access_floor`` are deleted. The defaults
           (30 days, 0 accesses) mean "never retrieved in 30 days → gone."

        Working memory is untouched — it lives in process and evicts via
        FIFO already.

        Args:
            now: Reference time. Defaults to ``datetime.now(UTC)``. Pass
                explicitly in tests.
            ttl_only: If ``True``, skip phase 2 and only delete expired
                records. Useful when you want hard-deadline semantics
                without the heuristic cold-eviction.
            access_floor: Records with ``access_count <= access_floor``
                are eligible for cold eviction. Default 0 — anything ever
                retrieved survives.
            cold_age_days: A record must be at least this old (by
                ``created_at``) to be considered cold.
            max_per_tier: Hard cap on records scanned per tier per call.
                Forgetting is O(records scanned). On a backend with
                millions of records, raise the cap or run forget() more
                often with a smaller window.

        Returns:
            ``{"expired": N, "cold": M}`` so callers can audit and log.
        """
        reference_time = now if now is not None else datetime.now(UTC)
        cold_threshold = reference_time - timedelta(days=cold_age_days)

        expired_count = 0
        cold_count = 0

        for tier in _FORGETTABLE_TIERS:
            records = self.backend.list_recent(
                agent_id=self.agent_id,
                tier=tier,
                limit=max_per_tier,
            )
            for record in records:
                # Phase 1: TTL.
                if record.expires_at is not None and record.expires_at <= reference_time:
                    if self.backend.delete(record.id):
                        expired_count += 1
                    continue
                # Phase 2: access-frequency decay (skipped on ttl_only).
                if ttl_only:
                    continue
                if (
                    record.created_at <= cold_threshold
                    and record.access_count <= access_floor
                    and self.backend.delete(record.id)
                ):
                    cold_count += 1

        return {"expired": expired_count, "cold": cold_count}

    # ---------------------------------------------------------------------
    # Scheduled cadence (v0.3)
    # ---------------------------------------------------------------------

    def start_scheduler(
        self,
        *,
        consolidate_every: timedelta | None = timedelta(minutes=30),
        forget_every: timedelta | None = timedelta(days=1),
        forget_kwargs: dict[str, Any] | None = None,
        consolidate_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Start a background scheduler running consolidation + forgetting.

        Requires the ``[scheduler]`` extra (``pip install 'mneme[scheduler]'``).
        Spins up an APScheduler ``BackgroundScheduler`` in a daemon thread
        and registers up to two periodic jobs.

        Idempotent: calling :meth:`start_scheduler` while one is already
        running raises ``RuntimeError`` — stop the existing scheduler first.

        Args:
            consolidate_every: Period for :meth:`consolidate`. ``None`` skips
                the consolidation job (useful when you handle it yourself).
                Default 30 minutes.
            forget_every: Period for :meth:`forget`. ``None`` skips the
                forgetting job. Default daily.
            consolidate_kwargs: Keyword args forwarded to each
                ``consolidate()`` call.
            forget_kwargs: Keyword args forwarded to each ``forget()`` call.

        Raises:
            ImportError: if the ``[scheduler]`` extra isn't installed.
            RuntimeError: if a scheduler is already running, or if
                ``consolidate_every`` is set but ``self.llm_judge`` is None.
        """
        if self._scheduler is not None:
            raise RuntimeError("A scheduler is already running. Call stop_scheduler() first.")
        if consolidate_every is not None and self.llm_judge is None:
            raise RuntimeError(
                "Cannot schedule consolidation without an llm_judge. "
                "Either pass llm_judge= to MemoryManager(...) or set "
                "consolidate_every=None."
            )

        try:
            from apscheduler.schedulers.background import (
                BackgroundScheduler,
            )
        except ImportError as exc:
            raise ImportError(
                "start_scheduler requires the 'scheduler' extra. Install with:\n"
                "    pip install 'mneme[scheduler]'\n"
                "    # or, in a uv-managed project:\n"
                "    uv add mneme --extra scheduler"
            ) from exc

        scheduler = BackgroundScheduler(daemon=True)

        ckw = consolidate_kwargs or {}
        fkw = forget_kwargs or {}

        if consolidate_every is not None:
            scheduler.add_job(
                lambda: self.consolidate(**ckw),
                trigger="interval",
                seconds=consolidate_every.total_seconds(),
                id="mneme_consolidate",
                replace_existing=True,
                max_instances=1,
                coalesce=True,
            )
        if forget_every is not None:
            scheduler.add_job(
                lambda: self.forget(**fkw),
                trigger="interval",
                seconds=forget_every.total_seconds(),
                id="mneme_forget",
                replace_existing=True,
                max_instances=1,
                coalesce=True,
            )

        scheduler.start()
        self._scheduler = scheduler

    def stop_scheduler(self, *, wait: bool = True) -> None:
        """Stop the background scheduler if one is running. Idempotent.

        Args:
            wait: Forwarded to APScheduler's ``shutdown(wait=...)``. If
                ``True`` (default), blocks until in-flight jobs finish.
        """
        scheduler = self._scheduler
        self._scheduler = None
        if scheduler is None:
            return
        # If the scheduler was already shut down externally we don't want
        # stop_scheduler() itself to raise. Idempotency wins.
        with contextlib.suppress(Exception):
            scheduler.shutdown(wait=wait)

    def __del__(self) -> None:
        # Best-effort safety net so a forgotten stop_scheduler() doesn't leave
        # a thread dangling past the manager's lifetime.
        if getattr(self, "_scheduler", None) is not None:
            with contextlib.suppress(Exception):
                self.stop_scheduler(wait=False)

    # ---------------------------------------------------------------------
    # Maintenance
    # ---------------------------------------------------------------------

    def clear_all(self) -> dict[str, int]:
        """Wipe every memory this agent has, across every tier.

        Returns a dict ``{tier_name: count_deleted}`` so callers can audit.
        Useful for tests and for explicit "forget this agent" operations.
        """
        return {
            "working": self.working.clear(),
            "episodic": self.episodic.clear(),
            "semantic": self.semantic.clear(),
        }


# ---------------------------------------------------------------------------
# Score components — module-level helpers for testability + clarity.
# ---------------------------------------------------------------------------


def _recency_decay(
    *,
    created_at: datetime,
    now: datetime,
    half_life_days: float,
) -> float:
    """Exponential freshness decay.

    Returns ``0.5 ** (age_days / half_life_days)``. A just-created record
    scores 1.0; a half-life-old record scores 0.5; an indefinitely-old
    record decays toward 0.

    A *future* ``created_at`` (clock skew, manual backdating) returns 1.0
    rather than a value > 1, so adversarial timestamps cannot boost a
    record's recency above "brand new."
    """
    if half_life_days <= 0:
        raise ValueError(f"half_life_days must be positive, got {half_life_days}")
    age_seconds = (now - created_at).total_seconds()
    if age_seconds <= 0:
        return 1.0
    age_days = age_seconds / 86400.0
    # ``math.pow`` is used instead of the ``**`` operator because mypy strict
    # can't narrow the result of ``float ** float`` through Python's pow
    # overloads — ``math.pow`` has a clean ``(float, float) -> float`` signature.
    return math.pow(0.5, age_days / half_life_days)


# ---------------------------------------------------------------------------
# Consolidation helpers (v0.2)
# ---------------------------------------------------------------------------


def _format_episode_for_prompt(episode: EpisodicMemory) -> str:
    """Render a single episode as the consolidator's prompt expects it."""
    return f"[id={episode.id} created_at={episode.created_at.isoformat()}] {episode.content}"


def _extract_facts_from_batch(
    *,
    judge: LLMJudge,
    episodes: list[EpisodicMemory],
) -> list[dict[str, Any]]:
    """Call the judge with the extraction prompt and return the raw fact dicts.

    Each dict is shaped per :data:`_FACT_EXTRACTION_SCHEMA`: keys ``content``
    (str), ``confidence`` (float), ``source_episode_ids`` (list[str]).
    The OpenAI API enforces this server-side when running strict json_schema;
    mock judges should also conform if they want to exercise the happy path.
    """
    if not episodes:
        return []

    rendered = "\n".join(_format_episode_for_prompt(e) for e in episodes)
    user_message = (
        "Extract durable facts from the following episodes. Return JSON "
        f"matching the provided schema.\n\nEpisodes:\n{rendered}"
    )
    response = judge.complete(
        messages=[
            {"role": "system", "content": _CONSOLIDATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        response_schema=_FACT_EXTRACTION_SCHEMA,
        schema_name="FactExtraction",
        temperature=0.0,
    )
    raw_facts = response.get("facts", [])
    # The judge is responsible for schema conformance; if a buggy mock returns
    # something off-shape we'd rather raise loudly here than corrupt storage.
    if not isinstance(raw_facts, list):
        raise RuntimeError(f"LLM judge returned non-list 'facts' field: {raw_facts!r}")
    return list(raw_facts)


def _store_or_merge_fact(
    *,
    semantic: SemanticMemoryTier,
    embedder: EmbeddingProvider,
    candidate: dict[str, Any],
    dedup_threshold: float,
) -> SemanticFact:
    """Persist one extracted fact, merging into an existing one if similar enough.

    Merge policy is *newer wins with provenance preserved*:

    * If the most-similar existing fact has cosine similarity ≥
      ``dedup_threshold``, replace its content with the candidate's content,
      union the provenance lists, take ``max`` of the two confidences, and
      keep the existing id.
    * Otherwise, write the candidate as a brand-new :class:`SemanticFact`.

    Returns the fact that was created or updated.
    """
    content = str(candidate["content"])
    confidence = float(candidate["confidence"])
    provenance = [str(x) for x in candidate.get("source_episode_ids", [])]

    # Compare the candidate against existing semantic facts via similarity
    # search. Top-1 is sufficient — if anything is going to merge, it's the
    # nearest neighbour.
    similar = semantic.search(content, k=1)
    if similar:
        existing, similarity = similar[0]
        if similarity >= dedup_threshold:
            existing.content = content
            existing.confidence = max(existing.confidence, confidence)
            # Union with order-preserving dedup so the call site stays
            # predictable for tests.
            seen = set(existing.provenance)
            for pid in provenance:
                if pid not in seen:
                    existing.provenance.append(pid)
                    seen.add(pid)
            # Re-embed the updated content so future searches reflect it.
            [new_embedding] = embedder.embed([content])
            existing.embedding = new_embedding
            semantic.backend.upsert(existing)
            return existing

    return semantic.add(
        content,
        confidence=confidence,
        provenance=provenance,
    )
