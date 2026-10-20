# Architectural & Technical Decisions

This file documents every meaningful technical decision made in this project, with the reasoning behind each choice.

---

## 1. Overall Architecture: Hybrid Deep + Gradient Boosting

**Decision:** Use two deep autoencoders (TransformerVAE + LSTM) purely as *feature extractors*, feeding their per-channel reconstruction errors into an XGBoost classifier as the final anomaly scorer.

**Why not use the deep models directly for scoring?**
Deep autoencoders trained only on normal data produce high reconstruction error on anomalous inputs — that is the detection signal. But raw MSE thresholds are fragile: they don't naturally incorporate vessel-specific baselines, static vessel metadata (type, length, beam), or kinematic statistics derived from the 30-point window. XGBoost can learn non-linear combinations of all 55 features at once, including those contextual signals.

**Why two deep models instead of one?**
TransformerVAE captures long-range temporal dependencies via self-attention; LSTM Autoencoder captures local sequential patterns via recurrence. Their reconstruction errors are complementary — one may flag a pattern the other misses. XGBoost learns which model to trust per feature channel.

**Why XGBoost as the final classifier?**
- Fast, deterministic inference (no stochastic sampling)
- Handles mixed feature types (continuous kinematics + discrete vessel type)
- Natively outputs a probability score in [0, 1], easy to threshold
- Can be trained on a relatively small labelled dataset (per-incident-type labelled sequences) without overfitting

---

## 2. FastAPI as the Serving Framework

**Decision:** Serve predictions via a FastAPI app (`mlops/main.py`).

**Why FastAPI over Flask or Django?**
- Native `async/await` support pairs naturally with `asyncio`-based consumers
- Pydantic-powered request/response validation is built in — no manual schema code
- Automatic OpenAPI/Swagger docs at `/docs` with zero extra config
- `lifespan` context manager gives a clean startup hook for loading models once at boot, not per-request

**Why not gRPC or a message-queue consumer?**
Single-client demo scope. REST over HTTP is the simplest interface that a live AIS consumer (`ais_consumer.py`) can POST to without needing shared memory or a broker.

---

## 3. Pydantic for Input Validation

**Decision:** Every incoming AIS point is parsed through a `AISPoint` Pydantic model with field-level validators before touching any ML code.

**Why?**
AIS data is messy. Vessels broadcast sentinel values (`SOG=102.3` means "not available", `COG=360.0` means "not available", `lat=91`, `lon=181`). Without explicit rejection, these sentinels would silently corrupt feature computation. `field_validator` rejects them at the HTTP boundary, keeping all downstream code clean.

**Why strip `input` and `ctx` from Pydantic's 422 error payload?**
Pydantic's default `RequestValidationError` response echoes the raw input value (which can be `NaN` or `Infinity`) and a `ctx` dict containing the raw Python `Exception` object — neither is JSON-serialisable. The custom exception handler (`_validation_error_handler`) strips those keys so the API never returns a 500 due to a bad 422.

---

## 4. SQLite for the Vessel Profile Store

**Decision:** Persist per-vessel baselines in a local SQLite database (`vessel_profiles.db`).

**Why SQLite over Redis, PostgreSQL, or a file?**
- No extra service to run — the entire stack is one container
- Adequate for a single-worker, single-client demo (SQLite's write lock is not a bottleneck at ~1 write/min per vessel)
- Python's `sqlite3` is in the standard library — no extra dependency
- Survives process restarts (vessel baselines accumulate across runs)

**Known limitation documented in code:** SQLite's connection-per-call pattern is not concurrency-safe for multiple FastAPI workers targeting the same MMSI. For multi-worker production, replace with Redis or a Postgres row-level lock.

---

## 5. In-Memory Per-Vessel Rolling Buffers

**Decision:** `_vessel_buffers` and `_last_raw_point` in `feature_factory.py` are plain Python dicts, kept in the FastAPI process's memory.

**Why not persist the buffer in SQLite?**
Each buffer entry holds 7 floating-point enriched kinematic values. Reading and writing 30 rows per prediction from SQLite would add latency without benefit — the buffer is transient by design and can be rebuilt by replaying AIS messages after a restart. The vessel *profile* (aggregated statistics) is what must survive restarts, and it is in SQLite.

**Known limitation:** Multiple FastAPI workers would each maintain their own partial buffer, producing incorrect windowed features for the same vessel. For production horizontal scaling, replace with a Redis LPUSH/LTRIM-backed ring buffer.

---

## 6. Window Size = 30 Points

**Decision:** Both deep models and the XGBoost feature extractor operate on a fixed 30-point sliding window.

**Why 30?**
The training data uses AIS at ~60–90 s intervals. A 30-point window covers ~40 minutes of vessel trajectory — long enough to capture meaningful behavioural patterns (sustained course deviation, speed collapse, entry into a restricted zone) while short enough to make real-time scoring practical. The window was the same size used during all training runs, so changing it at serving time would invalidate the loaded model weights.

---

## 7. Downsampling AIS Position Reports to 60 s Minimum Interval

**Decision:** `PositionThrottle` in `ais_consumer.py` drops position messages for a vessel if the gap since its last accepted message is less than `MIN_POSITION_INTERVAL_SEC` (default 60 s).

**Why?**
Class A transponders broadcast every 2–10 s when a ship is under way. At that rate, a 30-point buffer fills in under 5 minutes. The XGBoost model's `delta_time_mean` feature has splits starting at ~65 s — it was trained on the historical dataset sampled at ~60–90 s. Feeding it 5-second intervals would push `delta_time_mean` far outside the training distribution and make scores meaningless.

---

## 8. Clamping Outliers in Serving (Not Dropping)

**Decision:** When a kinematic value (computed speed, acceleration, heading change) falls outside the training-time filter bounds, it is **clamped** to the bound rather than dropped.

**Why not drop?**
Dropping a point mid-stream would leave a gap in the vessel's rolling buffer. The very next point would compute deltas against the previous *kept* point, making `delta_time` appear much larger than it actually is. Clamping preserves continuity; the clamp is documented as a known training/serving discrepancy.

---

## 9. Deterministic Deep Model Scoring (encode → mu → decode)

**Decision:** At inference time, `_deterministic_transformer_score` calls `model.encode(x)` to get `mu`, then `model.decode(mu)` directly — bypassing `reparameterize()`.

**Why?**
`reparameterize()` adds `eps * std` where `eps ~ N(0,1)` — the result is different every call. The reconstruction error used as an XGBoost feature would then be non-deterministic, making regression tests impossible and model behaviour unpredictable. Using `mu` (the mean of the latent distribution) gives the same output every run.

This was discovered during per-incident-type diagnostic work where identical inputs produced different anomaly scores across runs.

---

## 10. Anomaly Threshold = 0.85

**Decision:** `ANOMALY_THRESHOLD = 0.85` in `main.py`. `is_anomaly=True` when XGBoost score ≥ 0.85.

**Why 0.85?**
This is the P95 of the XGBoost score distribution over the clean-2020 negative pool. At this threshold, 5% of genuinely normal windows are flagged — an acceptable false positive rate for a monitoring use case where operators review alerts. A lower threshold increases recall at the cost of alert fatigue.

---

## 11. Baseline Update Threshold = 0.77

**Decision:** `SCORE_UPDATE_THRESHOLD = 0.77` in `vessel_profile_store.py`. A point is added to the vessel's baseline sample only if its window score is below 0.77.

**Why not 0.85 (the detection threshold)?**
The gap between 0.77 and 0.85 creates a "grey zone" — points that are elevated but not yet flagged as anomalies are excluded from the baseline. This prevents a vessel drifting into mildly anomalous behaviour from slowly normalising its own profile and escaping detection.

---

## 12. Minimum 50 Points Before Vessel-Specific Baseline

**Decision:** `MIN_BASELINE_POINTS = 50`. Below this, `get_profile()` returns population-level fallback statistics.

**Why 50?**
At 60 s intervals, 50 points = ~50 minutes of observed data. Less than that and the computed mean/std of speed and heading are too noisy to be meaningful for z-score features. The population fallback (from training data medians) is more stable than a 5-point vessel estimate.

---

## 13. Population and Static Fallbacks from Training Data

**Decision:** Two JSON fallback files:
- `pop_fallback.json` — population-level speed, heading, and position statistics
- `static_fallback.json` — median vessel type code, length, and beam

**Why bake these into JSON files rather than hardcoding them?**
They are computed from the training dataset and would change if the training data changes. Storing them separately makes it easy to update them without modifying code. The model was trained with these exact values for cold-start vessels, so using them at serving time keeps the feature distribution consistent with training.

---

## 14. WebSockets + httpx for the AIS Consumer

**Decision:** `ais_consumer.py` connects to `wss://stream.aisstream.io/v0/stream` using the `websockets` library, and posts predictions to the local API using `httpx.AsyncClient`.

**Why `websockets` over `aiohttp` or `tornado`?**
`websockets` is a purpose-built, pure-Python async WebSocket client with explicit support for `permessage-deflate` compression (required by aisstream.io from September 2026) via a single argument: `compression="deflate"`. It integrates naturally with `asyncio`.

**Why `httpx` over `requests` for posting?**
`requests` is synchronous and would block the event loop. `httpx.AsyncClient` is a drop-in async equivalent that fits the `asyncio`-based consumer architecture without needing a separate thread pool.

---

## 15. asyncio.Queue for Backpressure Between WS Reader and API Workers

**Decision:** The WebSocket reader puts position payloads onto an `asyncio.Queue(maxsize=QUEUE_MAXSIZE)`. `WORKER_COUNT` async tasks drain the queue and POST to `/predict`. If the queue is full, the incoming payload is dropped (not awaited).

**Why not await `queue.put()` (which would block the reader)?**
Blocking the reader would cause the WebSocket server to see no reads and eventually close the connection. Dropping is preferable to backpressure that kills the stream. `stats.dropped` counts how often this happens.

**Why separate workers instead of posting directly from the reader?**
Each HTTP POST takes up to `REQUEST_TIMEOUT_SEC` (5 s). Without workers, a slow API would throttle the reader at the WebSocket frame level. Multiple workers (default 3) let the reader run ahead while workers drain at their own pace.

---

## 16. Exponential Backoff for WebSocket Reconnection

**Decision:** On any WebSocket error, the consumer waits `backoff` seconds before reconnecting, doubling up to `MAX_RECONNECT_SEC` (60 s).

**Why?**
Immediate reconnection loops can DDOS the upstream broker or get the client rate-limited/banned. Exponential backoff is the standard pattern for transient network failures.

**Auth fail-fast:** After `MAX_UNCONFIRMED_CONNECTIONS` (3) consecutive connections that close before a `SubscriptionConfirmation` arrives, the consumer exits rather than looping forever. This is how aisstream.io signals an invalid API key (it doesn't send an HTTP 401 — it just drops the connection).

---

## 17. Hawaii Bounding Box

**Decision:** The default bounding box covers the main Hawaiian Islands (`[[18.5, -161.0], [22.8, -154.5]]`).

**Why only Hawaii?**
All training data was sourced from Hawaiian AIS records (2017–2020). The latitude and longitude features in XGBoost are calibrated to this region. The centroid fallback (`lat ~21.3, lon ~-157.8`) is Honolulu harbour. Scoring vessels in a different ocean would push lat/lon features completely out of distribution and produce meaningless scores.

---

## 18. MMSI-Grouped Train/Val/Test Split

**Decision:** In `train_xgboost_hybrid.py`, the 70/15/15 split is done **by MMSI** (vessel identity), not by row.

**Why?**
A single vessel's 30-point windows overlap heavily (stride=15). Splitting by row would put windows from the same vessel in both training and test sets — the model would "memorise" individual vessel behaviour rather than generalising. Grouping by MMSI ensures no vessel appears in more than one split.

---

## 19. TransformerVAE Architecture Choices

**Decision:** `d_model=256`, `nhead=8`, `num_layers=6`, `latent_dim=64`, `dim_feedforward=512`.

**Why a VAE instead of a plain autoencoder?**
The VAE's KL divergence term regularises the latent space, preventing the encoder from memorising individual training sequences. A regularised latent space means the decoder's reconstruction error is a better proxy for "how far is this sequence from the normal manifold" than a plain AE where the encoder can learn arbitrary bijections.

**Why use the Transformer architecture for sequence modelling?**
AIS trajectories have long-range correlations: a vessel's heading 20 minutes ago is relevant to detecting whether its current position is plausible. Self-attention can directly model any position-to-position relationship in the 30-step window, while an LSTM's gradients decay exponentially over distance.

---

## 20. Logging Strategy

**Decision:** All modules use Python's standard `logging` with structured key=value pairs (e.g., `"consumer_stats received=%d ..."`).

**Why structured log lines over f-strings with prose?**
Key=value log lines can be parsed by log aggregators (Datadog, Loki, ELK) without regex. Every log event can be filtered by field (e.g., `mmsi=123456789 is_anomaly=True`). This was chosen over a JSON logger to keep the dependency list minimal while still being grep-friendly.

---

## 21. `geopy.distance.distance` for Computed Speed

**Decision:** Geographic distance between consecutive AIS points is computed using `geopy.distance.distance` (Vincenty formula), not the Haversine approximation.

**Why Vincenty over Haversine?**
At the scales relevant to maritime navigation (vessels travelling tens of kilometres between AIS points), Haversine introduces errors of ~0.3% due to Earth's oblateness. Vincenty is accurate to millimetres. Since `computed_speed_knots` is a trained feature, any systematic distance bias would shift the feature distribution relative to training time.

---

## 22. `numpy` for Feature Aggregation, `torch` for Deep Models

**Decision:** Kinematic statistics (mean, std, min, max over the 30-point window) are computed with `numpy`. Deep model forward passes use `torch`.

**Why not use `torch` tensors for everything?**
`numpy` operations on small arrays (30 elements) are faster to set up than equivalent `torch` operations because there is no GPU dispatch overhead for CPU tensors. The XGBoost `DMatrix` constructor also expects a `numpy` array. `torch` is reserved for the batched forward passes where its batching and GPU acceleration are actually beneficial.

---

## 23. `pytest-asyncio` for Async Test Coverage

**Decision:** The test suite uses `pytest-asyncio` to test async functions in `ais_consumer.py` without spinning up a real WebSocket server.

**Why?**
`ais_consumer.py` has significant async logic (queue draining, reconnection, throttling). Testing it synchronously would require rewriting the code under test. `pytest-asyncio` lets tests `await` async functions directly, keeping tests simple and the production code unmodified.
