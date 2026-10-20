# Code Execution Flow

This file documents how execution travels through the codebase: entry points, call chains, and the order in which files and functions are invoked.

There are three independent execution paths in this project:

1. **Offline Training Pipeline** — run once to produce model artifacts
2. **Online Serving API** — `mlops/main.py`, the FastAPI server
3. **Live AIS Consumer** — `mlops/ais_consumer.py`, the WebSocket ingestion loop

---

## Path 1: Offline Training Pipeline

This path is run sequentially, producing the `.npy`, `.pt`, and `.json` artifacts that the serving layer loads.

```
processing/create_sequences.py
    │
    │  Reads raw AIS parquet files (2017–2020).
    │  Groups by MMSI, slides 30-point windows (stride=15).
    │  Writes: normal_sequences_{year}.npy  (shape: N×30×7)
    │
    ▼
processing/normalize(3).py
    │
    │  Loads normal_sequences_*.npy, concatenates all years.
    │  Computes mean and std per channel over the training split.
    │  Writes: processing/norm_mean.npy, processing/norm_std.npy
    │  Normalises all splits and writes X_train.npy, X_val.npy,
    │  X_normal_test_norm.npy, X_anomaly_test_norm.npy
    │
    ▼
model/train_translarge_ae.py  (or train_transsmall.py)
    │
    │  Loads X_train.npy, X_val.npy.
    │  Instantiates TransformerVAE (model/model.py::TransformerVAE).
    │  Trains with vae_loss (recon MSE + KL divergence).
    │  Writes: model/best_model_large.pt
    │
    ▼
model/train_lstm_ae.py
    │
    │  Loads X_train.npy, X_val.npy.
    │  Instantiates LSTMAutoencoder (defined inline in this file).
    │  Trains with MSE loss, early stopping on val loss.
    │  Writes: model/lstm_ae_run/lstm_ae_best.pt
    │
    ▼
hybrid/add_deep_model_scores.py
    │
    │  Loads both saved models (best_model_large.pt, lstm_ae_best.pt).
    │  Runs forward passes on all sequences (incident + clean).
    │  Computes per-channel reconstruction error for each 30-point window.
    │  Writes: hybrid/hybrid_features_*_with_scores.parquet
    │
    ▼
hybrid/train_xgboost_hybrid.py
    │
    │  Loads hybrid_features_*_with_scores.parquet.
    │  Performs MMSI-grouped 70/15/15 train/val/test split.
    │  Trains XGBoost on 55 features (kinematic stats + deep errors + profile).
    │  Writes: hybrid/xgboost_hybrid_final.json
    │
    ▼
  ARTIFACTS READY FOR SERVING:
    model/best_model_large.pt
    model/lstm_ae_run/lstm_ae_best.pt
    hybrid/xgboost_hybrid_final.json
    processing/norm_mean.npy
    processing/norm_std.npy
    pop_fallback.json
    static_fallback.json
```

---

## Path 2: Online Serving API

### Entry Point

```
uvicorn mlops.main:app --host 0.0.0.0 --port 8000
```

or directly:

```
python mlops/main.py
```

### Startup Sequence

```
mlops/main.py
    │
    │  FastAPI app is created.
    │  lifespan() async context manager runs at startup:
    │
    ├─► vessel_profile_store.init_db()
    │       Opens (or creates) vessel_profiles.db.
    │       Runs CREATE TABLE IF NOT EXISTS for:
    │           vessel_profile
    │           vessel_sample_points
    │       Creates indexes on (mmsi, observed_at) and (mmsi, id).
    │
    └─► feature_factory._get_resources()
            Acquires _resources_lock (threading.Lock).
            Instantiates TransformerVAE (model/model.py).
            Loads state dict from model/best_model_large.pt.
            Sets model to eval().
            Instantiates LSTMAutoencoder (model/train_lstm_ae.py).
            Loads state dict from model/lstm_ae_run/lstm_ae_best.pt.
            Sets model to eval().
            Instantiates xgb.Booster().
            Loads hybrid/xgboost_hybrid_final.json.
            Loads processing/norm_mean.npy, processing/norm_std.npy.
```

### Per-Request Flow: POST /predict

```
HTTP POST /predict  { MMSI, lat, lon, speed_over_ground_knots,
                      course_over_ground_deg, datetime_hst }
    │
    │  Pydantic parses and validates AISPoint:
    │      validate_lat()         — finite, in [-90, 90]
    │      validate_lon()         — finite, in [-180, 180]
    │      validate_sog()         — finite, in [0, 102.4]
    │      validate_cog()         — finite, in [0, 360]
    │      validate_mmsi()        — non-empty, ≤ 64 chars
    │  On failure → _validation_error_handler() → 422 (safe JSON)
    │
    ▼
mlops/main.py::predict()
    │
    ├─► feature_factory.process_ais_point(raw)
    │       │
    │       │  Normalises field names (MMSI→mmsi, speed/course keys).
    │       │  Parses timestamp:
    │       │      kinematics.parse_timestamp()
    │       │          datetime.fromisoformat() or pass-through.
    │       │
    │       ├─► _compute_deltas(mmsi, raw_point)
    │       │       Looks up _last_raw_point[mmsi] (previous raw point).
    │       │       Updates _last_raw_point[mmsi] = current raw point.
    │       │       If no previous point → returns None (first point ever).
    │       │       If dt_sec < 1 s → returns None (duplicate/out-of-order).
    │       │       Computes:
    │       │           dist_km  via geopy.distance.distance()
    │       │           computed_speed = (dist_km / dt_hrs) / 1.852
    │       │           acceleration   = (sog_now - sog_prev) / dt_sec
    │       │           heading_change via kinematics.angular_diff_deg()
    │       │       Clamps each to training-time bounds via kinematics.clamp().
    │       │       Returns enriched dict {lat, lon, sog, cog, computed_speed,
    │       │                              acceleration, heading_change, delta_time}.
    │       │
    │       │  If enriched is None → returns {ready: False}  (no score yet)
    │       │
    │       ├─► _add_enriched_point(mmsi, enriched)
    │       │       Appends to _vessel_buffers[mmsi] (creates list if new).
    │       │       Trims to MAX_BUFFER (60) entries.
    │       │
    │       │  If buffer length < WINDOW_SIZE (30) → returns {ready: False}
    │       │
    │       ├─► _extract_features(mmsi, window[-30:])
    │       │       Computes numpy arrays for each of the 7 kinematic channels.
    │       │       Computes per-channel mean/std/min/max → 28 stats.
    │       │       Computes delta_time mean/max/std → 3 stats.
    │       │       Calls vessel_profile_store.get_profile(mmsi):
    │       │           Queries vessel_profile table in SQLite.
    │       │           If missing or point_count < 50:
    │       │               Returns population fallback + static fallback.
    │       │           Else:
    │       │               Returns vessel-specific aggregates.
    │       │       Computes speed_zscore and heading_zscore vs vessel profile.
    │       │       Computes dist_from_vessel_centroid_km via geopy (×30 points).
    │       │       Calls _deterministic_transformer_score(buffer_raw):
    │       │           Normalises (buffer_raw - norm_mean) / (norm_std + 1e-8).
    │       │           Runs TransformerVAE.encode(x) → mu, logvar.
    │       │           Runs TransformerVAE.decode(mu) → recon.  ← NO reparameterize
    │       │           Computes (recon - x)² mean over time → per-channel MSE.
    │       │       Calls _deterministic_lstm_score(buffer_raw):
    │       │           Same normalisation.
    │       │           Runs LSTMAutoencoder.forward(x) → recon, latent.
    │       │           Computes per-channel MSE.
    │       │       Assembles 55-length feature vector in FEATURE_NAMES order.
    │       │
    │       ├─► _score(feature_vector)
    │       │       Creates xgb.DMatrix(feature_vector, feature_names=FEATURE_NAMES).
    │       │       Calls xgb_booster.predict(dmat) → float in [0, 1].
    │       │
    │       └─► vessel_profile_store.maybe_update_profile(mmsi, point, score)
    │               If score >= 0.77 → return immediately (anomalous/ambiguous)
    │               INSERT OR IGNORE vessel_profile row.
    │               INSERT into vessel_sample_points.
    │               _evict_oldest_if_over_cap(mmsi, cap=2000):
    │                   DELETE oldest rows if count > 2000.
    │               _recompute_aggregates(mmsi):
    │                   SELECT all sample points for vessel.
    │                   Compute mean/std of speed, heading; median of lat/lon.
    │                   UPDATE vessel_profile row.
    │
    ├─► vessel_profile_store.get_profile(result["mmsi"])
    │       (second call, to read confidence and baseline_is_fallback for response)
    │
    ├─► feature_factory.get_buffer_length(result["mmsi"])
    │       len(_vessel_buffers.get(mmsi, []))
    │
    └─► Returns PredictionResponse:
            mmsi, anomaly_score, confidence, is_anomaly,
            baseline_established, point_count
```

### Health / Readiness Endpoints

```
GET /health  →  {"status": "ok"}       (always, no logic)
GET /ready   →  {"status": "ready"}    (always, no logic)
```

---

## Path 3: Live AIS Consumer

### Entry Point

```
python -m mlops.ais_consumer
  or
AISSTREAM_API_KEY=<key> python mlops/ais_consumer.py
```

### Startup Sequence

```
ais_consumer.main()
    │
    └─► asyncio.run(_async_main())
            │
            ├─► _setup_logging()
            │       logging.basicConfig(level=INFO, format=...)
            │
            ├─► signal handlers: SIGINT / SIGTERM → cancel main_task
            │
            └─► run_consumer()
```

### run_consumer() — Full Lifecycle

```
run_consumer()
    │
    ├─► load_watchlist()
    │       Check WATCHLIST_MMSIS env → parse_mmsis(csv)
    │       Or read watchlist.json (or {"mmsis": [...]})
    │       → parse_mmsis(list)
    │       Returns list of 9-digit MMSI strings (max 200).
    │
    ├─► vessel_profile_store.init_db()   (idempotent; creates tables if missing)
    │
    ├─► asyncio.Queue(maxsize=500)
    │
    ├─► httpx.AsyncClient (one client, shared across reconnects)
    │
    ├─► spawn WORKER_COUNT (3) asyncio Tasks:
    │       Each runs _worker(queue, client, stats) forever:
    │           payload = await queue.get()
    │           await post_prediction(client, payload, stats)
    │               httpx.AsyncClient.post(PREDICT_URL, json=payload)
    │               On 422: log warning, increment server_errors.
    │               On ≥400: log error, increment server_errors.
    │               On success: if is_anomaly → log warning + increment anomalies.
    │               On timeout: log warning, increment server_errors.
    │           queue.task_done()
    │
    └─► Reconnect loop:
            while True:
                await consume_stream(queue, mmsis=watchlist)
                    │
                    │  websockets.connect(AISSTREAM_WS_URL, compression="deflate")
                    │  Send subscription JSON:
                    │      build_subscription(api_key, boxes, mmsis)
                    │          {"APIKey":..., "BoundingBoxes":...,
                    │           "FilterMessageTypes": [...], "FiltersShipMMSI": [...]}
                    │
                    │  async for raw_message in ws:
                    │      json.loads(raw_message) → data dict
                    │      msg_type = data["MessageType"]
                    │
                    │      if "SubscriptionConfirmation":
                    │          confirmed = True
                    │          continue
                    │
                    │      if "error"/"Error" key:
                    │          log error, continue
                    │
                    │      if msg_type in _STATIC_MSG_TYPES:
                    │          handle_static_message(msg_type, data)
                    │              if "ShipStaticData":
                    │                  ShipStaticDataMessage.model_validate(data)
                    │                  Extract MMSI, vessel_type, length, width.
                    │              elif "StaticDataReport":
                    │                  StaticDataReportMessage.model_validate(data)
                    │                  Only Part B (PartNumber=True, Valid≠False).
                    │              vessel_profile_store.update_static_attributes(mmsi, ...)
                    │                  INSERT OR IGNORE vessel_profile row.
                    │                  UPDATE vessel_type_code/length_m/width_m.
                    │          continue
                    │
                    │      if msg_type in _CLASS_B_TYPES:
                    │          as_position_report(data)
                    │              Renames msg_type to "PositionReport",
                    │              reshapes Message key to {"PositionReport": body}.
                    │
                    │      AISStreamMessage.model_validate(data) → msg
                    │
                    │      position_is_usable(msg.Message.PositionReport):
                    │          Rejects: Valid=False, SOG≥102.3, COG≥360,
                    │                   lat outside [-90,90], lon outside [-180,180].
                    │          If not usable → stats.invalid++, continue
                    │
                    │      _throttle.allow(mmsi, parse_timestamp(time_utc)):
                    │          PositionThrottle: reject if gap < 60 s.
                    │          If throttled → stats.throttled++, continue
                    │
                    │      translate_to_predict_payload(msg)
                    │          → {MMSI, lat, lon, speed_over_ground_knots,
                    │             course_over_ground_deg, datetime_hst}
                    │
                    │      queue.put_nowait(payload)
                    │          If full → stats.dropped++, log warning
                    │
                    On SubscriptionRejected (close before confirmation):
                        unconfirmed++
                        if unconfirmed >= 3 → exit (bad API key)
                        else → sleep(backoff), backoff *= 2
                    On InvalidStatus 401/403 → exit immediately
                    On WebSocketException / OSError → sleep(backoff), backoff *= 2
                    On CancelledError → stats.log(), return
```

### Shutdown Sequence

```
SIGINT / SIGTERM received
    │
    └─► main_task.cancel()
            consume_stream raises asyncio.CancelledError
            └─► run_consumer() finally block:
                    await asyncio.wait_for(queue.join(), timeout=10s)
                    for w in workers: w.cancel()
                    await asyncio.gather(*workers, return_exceptions=True)
```

---

## Module Dependency Map

```
mlops/main.py
    ├── mlops/feature_factory.py
    │       ├── mlops/kinematics.py          (clamp, angular_diff_deg, parse_timestamp)
    │       ├── mlops/vessel_profile_store.py (get_profile, maybe_update_profile)
    │       ├── model/model.py               (TransformerVAE)
    │       └── model/train_lstm_ae.py        (LSTMAutoencoder)
    └── mlops/vessel_profile_store.py

mlops/ais_consumer.py
    ├── mlops/kinematics.py                  (parse_timestamp)
    └── mlops/vessel_profile_store.py        (init_db, update_static_attributes)

mlops/discover_vessels.py
    └── mlops/ais_consumer.py                (reuses _POSITION_TYPES, AISStreamMessage,
                                              as_position_report, build_subscription)
```

---

## Key Global State (In-Process)

| Variable | Location | Lifetime | Purpose |
|---|---|---|---|
| `_vessel_buffers` | `feature_factory.py` | process lifetime | Rolling 60-point enriched buffer per MMSI |
| `_last_raw_point` | `feature_factory.py` | process lifetime | Last raw AIS point per MMSI, for delta computation |
| `_transformer_model` | `feature_factory.py` | process lifetime | Loaded TransformerVAE weights |
| `_lstm_model` | `feature_factory.py` | process lifetime | Loaded LSTMAutoencoder weights |
| `_xgb_booster` | `feature_factory.py` | process lifetime | Loaded XGBoost model |
| `_norm_mean` / `_norm_std` | `feature_factory.py` | process lifetime | Training normalisation statistics |
| `_throttle` | `ais_consumer.py` | process lifetime | Per-vessel last-seen timestamp for downsampling |
| `_stats` | `ais_consumer.py` | process lifetime | Running counters (received, posted, anomalies, etc.) |
| `vessel_profiles.db` | SQLite (disk) | persistent | Per-vessel baselines, static attributes |

---

## Data Shape Through the Serving Pipeline

```
Incoming JSON
  { MMSI: str, lat: float, lon: float, sog: float, cog: float, ts: str }
          │
          ▼ _compute_deltas()
  enriched point (dict, 8 fields)
  { lat, lon, sog, cog, computed_speed, acceleration, heading_change, delta_time }
          │
          ▼ _vessel_buffers[mmsi].append()
  rolling buffer: list of up to 60 such dicts
          │
          ▼ window = buf[-30:]  (30 dicts)
          │
          ▼ _extract_features()
  buffer_raw: np.ndarray (30, 7)  ← stacked kinematic channels
          │
          ├─► TransformerVAE: (1, 30, 7) → encode → mu (1, 64) → decode → (1, 30, 7)
          │   MSE per channel → trans_err[7] + trans_err_agg
          │
          ├─► LSTMAutoencoder: (1, 30, 7) → encode → latent (1, 64) → decode → (1, 30, 7)
          │   MSE per channel → lstm_err[7] + lstm_err_agg
          │
          └─► feature_vector: np.ndarray (55,)
                    │
                    ▼ xgb.Booster.predict()
              anomaly_score: float in [0, 1]
```
