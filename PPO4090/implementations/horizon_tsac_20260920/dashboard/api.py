"""Local-only by default, read-only REST/SSE with bounded queries and ID routing."""
import asyncio
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ..telemetry.storage import DB_NAME, MANIFEST_NAME, dumps, readonly, reconstruct
from ..audit import sha256


def create_app(runs_root=None, datasets_root=None, static_root=None):
    from project_paths import HORIZON_OUTPUT_DIR, HORIZON_DATA_DIR, HORIZON_FRONTEND_DIR
    root = Path(runs_root or HORIZON_OUTPUT_DIR).resolve()
    data_root = Path(datasets_root or (root if runs_root is not None else HORIZON_DATA_DIR)).resolve()
    assets = Path(static_root or HORIZON_FRONTEND_DIR / "dist").resolve()
    app = FastAPI(title="Horizon T-SAC 只读研究看板", version="1.0.0")
    integrity_cache = {}
    file_hash_cache = {}

    def discover():
        result = {}
        if root.exists():
            # All paths originate from our configured root, never client paths.
            for manifest_path in root.rglob(MANIFEST_NAME):
                if len(manifest_path.relative_to(root).parts) > 6:
                    continue
                folder = manifest_path.parent.resolve()
                if folder != root and root not in folder.parents:
                    continue
                if not (folder / DB_NAME).is_file():
                    continue
                if folder not in (folder / DB_NAME).resolve().parents or folder not in manifest_path.resolve().parents:
                    continue
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    identifier = str(manifest["run_id"])
                    if identifier not in result:
                        result[identifier] = (folder, manifest)
                except (ValueError, OSError, KeyError):
                    continue
        return result

    def resolve_run(run_id):
        found = discover().get(run_id)
        if found is None:
            raise HTTPException(404, "Run ID not found")
        return found

    @contextmanager
    def connection(run_id, location=None):
        path = location if location is not None else resolve_run(run_id)[0]
        conn = readonly(path / DB_NAME)
        conn.execute("BEGIN")
        try:
            yield conn
        finally:
            conn.rollback()
            conn.close()

    def watermark(conn):
        return conn.execute("SELECT COALESCE(MAX(event_seq),0) FROM events").fetchone()[0]

    def event_dict(row, manifest):
        result = dict(row)
        result["payload"] = json.loads(result["payload"])
        return dict(schema_version="horizon.telemetry.v1", run_id=manifest["run_id"],
                    trial_id=manifest.get("trial_id"), attempt_id=manifest.get("attempt_id"), **result)

    def audit_integrity(folder, artifacts):
        referenced = set()
        signatures, files = [], []
        for row in artifacts:
            path = (folder / row["path"]).resolve()
            safe = folder in path.parents
            exists = safe and path.is_file()
            stat = path.stat() if exists else None
            signatures.append((row["artifact_id"], row["sha256"], stat.st_size if stat else None, stat.st_mtime_ns if stat else None))
            referenced.add(path)
            files.append((row, path, safe, exists, signatures[-1]))
        snapshots = folder / "场景快照"
        orphans = sorted(p.relative_to(folder).as_posix() for p in snapshots.glob("*")
                         if p.is_file() and p.resolve() not in referenced) if snapshots.is_dir() else []
        signature = (tuple(signatures), tuple(orphans))
        cache = integrity_cache.get(str(folder))
        if cache and cache[0] == signature:
            return cache[1]
        failures = []
        for row, path, safe, exists, fingerprint in files:
            if not safe or not exists:
                failures.append({"artifact_id": row["artifact_id"], "reason": "unsafe_path" if not safe else "missing"})
                continue
            cache_key = (str(path), fingerprint)
            if cache_key not in file_hash_cache:
                try:
                    file_hash_cache[cache_key] = sha256(path)
                except OSError:
                    failures.append({"artifact_id": row["artifact_id"], "reason": "unreadable"})
                    continue
            if file_hash_cache[cache_key] != row["sha256"]:
                failures.append({"artifact_id": row["artifact_id"], "reason": "hash_mismatch"})
        result = {"status": "corrupted" if failures else "orphan_warning" if orphans else "verified",
                  "artifact_count": len(artifacts), "failures": failures, "unreferenced_snapshots": orphans,
                  "state_validation": "hash_checked_on_each_requested_state", "mutations_performed": False}
        integrity_cache[str(folder)] = (signature, result)
        return result

    @app.get("/api/v1/runs")
    def runs():
        items = []
        for run_id, (folder, manifest) in discover().items():
            with connection(run_id, folder) as conn:
                row = conn.execute("SELECT status FROM runs LIMIT 1").fetchone()
                last = conn.execute("SELECT committed_at_utc,event_type,elapsed_seconds FROM events ORDER BY event_seq DESC LIMIT 1").fetchone()
                heartbeat = conn.execute("SELECT occurred_at_utc FROM events WHERE event_type='performance' ORDER BY event_seq DESC LIMIT 1").fetchone()
                artifact_rows = conn.execute("SELECT * FROM artifacts").fetchall()
                item = {**manifest, "status": row[0], "watermark": watermark(conn),
                              "last_committed_at_utc": last[0] if last else None,
                              "last_heartbeat_at_utc": heartbeat[0] if heartbeat else None,
                              "last_event_type": last[1] if last else None}
            # Hashing never holds a read transaction open, and unchanged files
            # retain individual hash cache entries when new artifacts arrive.
            item["integrity"] = audit_integrity(folder, artifact_rows)
            items.append(item)
        return {"items": sorted(items, key=lambda x: x.get("created_at_utc", ""), reverse=True)}

    @app.get("/api/v1/runs/{run_id}/integrity")
    def integrity(run_id: str):
        folder, _ = resolve_run(run_id)
        with connection(run_id, folder) as conn:
            rows = conn.execute("SELECT * FROM artifacts").fetchall()
        return audit_integrity(folder, rows)

    @app.get("/api/v1/runs/{run_id}/manifest")
    def manifest(run_id: str):
        with connection(run_id) as conn:
            row = conn.execute("SELECT * FROM runs LIMIT 1").fetchone()
            return {**json.loads(row["manifest"]), "status": row["status"], "watermark": watermark(conn)}

    @app.get("/api/v1/runs/{run_id}/events")
    def events(run_id: str, cursor: int = Query(0, ge=0), limit: int = Query(200, ge=1, le=1000),
               severity: str = None, component: str = None, event_type: str = None, phase: str = None,
               episode_id: str = None, scenario_id: str = None, beam_id: str = None, text: str = Query(None, max_length=200)):
        where, args = ["event_seq> ?"], [cursor]
        for name, value in (("severity", severity), ("component", component), ("event_type", event_type), ("phase", phase),
                            ("episode_id", episode_id), ("scenario_id", scenario_id), ("beam_id", beam_id)):
            if value is not None:
                where.append(f"{name}=?")
                args.append(value)
        if text:
            where.append("instr(payload,?)>0")
            args.append(text)
        folder, run_manifest = resolve_run(run_id)
        with connection(run_id, folder) as conn:
            rows = conn.execute("SELECT * FROM events WHERE " + " AND ".join(where) + " ORDER BY event_seq LIMIT ?", [*args, limit]).fetchall()
            return {"items": [event_dict(row, run_manifest) for row in rows],
                    "next_cursor": rows[-1]["event_seq"] if rows else cursor, "watermark": watermark(conn)}

    @app.get("/api/v1/runs/{run_id}/episodes")
    def episodes(run_id: str, cursor: int = Query(0, ge=0), limit: int = Query(200, ge=1, le=1000)):
        with connection(run_id) as conn:
            rows = conn.execute("SELECT * FROM episodes WHERE initial_event_seq>? ORDER BY initial_event_seq LIMIT ?", (cursor, limit)).fetchall()
            return {"items": [{**dict(row), "summary": json.loads(row["summary"]) if row["summary"] else None} for row in rows],
                    "next_cursor": rows[-1]["initial_event_seq"] if rows else cursor, "watermark": watermark(conn)}

    @app.get("/api/v1/runs/{run_id}/episodes/{episode_id}/state")
    def state(run_id: str, episode_id: str, step: int = Query(..., ge=0)):
        with connection(run_id) as conn:
            return reconstruct(conn, episode_id, step)

    @app.get("/api/v1/runs/{run_id}/episodes/{episode_id}/transitions")
    def transitions(run_id: str, episode_id: str, from_step: int = Query(1, alias="from", ge=0), limit: int = Query(200, ge=1, le=1000)):
        with connection(run_id) as conn:
            rows = conn.execute("SELECT * FROM transitions WHERE episode_id=? AND step_index>=? ORDER BY step_index LIMIT ?", (episode_id, from_step, limit)).fetchall()
            return {"items": [{**dict(row), "transition": json.loads(row["transition"]), "delta": json.loads(row["delta"])} for row in rows],
                    "next_step": rows[-1]["step_index"] + 1 if rows else from_step, "watermark": watermark(conn)}

    @app.get("/api/v1/runs/{run_id}/metrics")
    def metrics(run_id: str, name: str = None, axis: str = "env_step", phase: str = None,
                start: float = None, end: float = None, limit: int = Query(500, ge=1, le=1000)):
        if axis not in ("env_step", "update_step", "episode", "elapsed_seconds"):
            raise HTTPException(422, "Unknown native axis")
        with connection(run_id) as conn:
            if name is None:
                rows = conn.execute("SELECT DISTINCT name,axis,phase FROM metrics ORDER BY axis,name").fetchall()
                return {"available": [dict(row) for row in rows], "items": [], "watermark": watermark(conn)}
            where, args = ["name=?", "axis=?"], [name, axis]
            for sql, value in (("phase=?", phase), ("axis_value>=?", start), ("axis_value<=?", end)):
                if value is not None:
                    where.append(sql)
                    args.append(value)
            clause = " AND ".join(where)
            total = conn.execute("SELECT COUNT(*) FROM metrics WHERE " + clause, args).fetchone()[0]
            # SQL aggregation bounds materialization even for long runs. Min/max and
            # anomaly counts survive; storage windows remain explicitly irreversible.
            if total <= limit:
                rows = conn.execute("SELECT * FROM metrics WHERE " + clause + " ORDER BY axis_value,metric_id", args).fetchall()
                items = [dict(row) for row in rows]
                aggregation = "stored_records"
            else:
                bucket = max(1, math.ceil(total / limit))
                rows = conn.execute("""WITH selected AS (SELECT *,ROW_NUMBER() OVER(ORDER BY axis_value,metric_id)-1 AS ordinal FROM metrics WHERE """ + clause + """)
                  SELECT phase, MIN(axis_value) AS first_axis,MAX(axis_value) AS last_axis,MAX(axis_value) AS axis_value,
                  SUM(count) AS count,SUM(value*count)/SUM(CASE WHEN value IS NOT NULL THEN count ELSE 0 END) AS value,
                  SUM(CASE WHEN value IS NULL THEN count ELSE 0 END) AS missing_count,MIN(min) AS min,MAX(max) AS max,
                  SUM(anomaly) AS anomaly,0 AS raw_available,'viewport' AS aggregation,
                  MIN(event_seq) AS first_event_seq,MAX(event_seq) AS event_seq
                  FROM selected GROUP BY CAST(ordinal/? AS INTEGER),phase ORDER BY first_axis""", [*args, bucket]).fetchall()
                items = [dict(row) for row in rows]
                aggregation = "min_max_count_viewport"
            return {"items": items, "name": name, "axis": axis, "count": total, "aggregation": aggregation,
                    "raw_available": all(bool(x.get("raw_available")) for x in items), "watermark": watermark(conn),
                    "units": "recorded_metric_name", "missing_value": "unrecorded_or_not_applicable"}

    @app.get("/api/v1/runs/{run_id}/stream")
    async def stream(request: Request, run_id: str, cursor: int = Query(0, ge=0)):
        folder, _ = resolve_run(run_id)
        header = request.headers.get("last-event-id")
        try:
            start_cursor = max(cursor, int(header)) if header else cursor
        except ValueError:
            raise HTTPException(422, "Invalid Last-Event-ID")
        async def generator():
            position = start_cursor
            while not await request.is_disconnected():
                with connection(run_id, folder) as conn:
                    high = watermark(conn)
                    rows = conn.execute("SELECT event_seq,event_id,event_type FROM events WHERE event_seq>? ORDER BY event_seq LIMIT 200", (position,)).fetchall()
                if position > high:
                    yield f"event: resync\ndata: {dumps({'watermark': high, 'reason': 'cursor_ahead'})}\n\n"
                    position = high
                for row in rows:
                    position = row["event_seq"]
                    yield f"id: {position}\nevent: committed\ndata: {dumps(dict(row))}\n\n"
                if not rows:
                    yield f": heartbeat {high}\n\n"
                await asyncio.sleep(.5)
        return StreamingResponse(generator(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/v1/comparisons")
    def comparisons(run_ids: str = Query(..., max_length=2000), phase: str = "test"):
        if phase not in ("validation", "test"):
            raise HTTPException(422, "Paired comparisons require a named evaluation phase")
        ids = list(dict.fromkeys(run_ids.split(",")))
        if len(ids) > 20:
            raise HTTPException(422, "At most 20 runs")
        rows = []
        keys = ("physics_version", "physics_config", "data_semantics", "metric_version", "dataset_version", "split_hash", "power_budget_w", "action_spec_signature")
        for run_id in ids:
            _, manifest = resolve_run(run_id)
            with connection(run_id) as conn:
                summaries = []
                for episode in conn.execute("SELECT scenario_id,summary FROM episodes WHERE status='completed' AND summary IS NOT NULL AND phase=?", (phase,)):
                    summary = json.loads(episode["summary"])
                    summaries.append({**summary, "scenario_id": episode["scenario_id"], "phase": phase})
                versions = manifest.get("versions", {})
                config = manifest.get("config", {})
                manifest = {**versions, **manifest}
                compatibility = {key: manifest.get(key) for key in keys}
                compatibility.update(physics_config=config.get("physics"), split_hash=manifest.get("split_hash", manifest.get("data_manifest_hash")))
                data_config = config.get("data")
                compatibility["data_semantics"] = ({key: data_config.get(key) for key in
                    ("source_schema", "rate_unit", "traffic_scale", "group_assignment", "service_order", "beamwidth_kind")}
                    if data_config is not None else None)
                rows.append({"run_id": run_id, "manifest": manifest, "compatibility": compatibility,
                             "status": conn.execute("SELECT status FROM runs").fetchone()[0], "episodes": summaries})
        differing = [key for key in keys if len({json.dumps(row["compatibility"].get(key), sort_keys=True) for row in rows}) > 1]
        missing = [key for key in keys if any(row["compatibility"].get(key) is None for row in rows)]
        return_compatible = len({row["manifest"].get("reward_version") for row in rows}) <= 1 and all(row["manifest"].get("reward_version") for row in rows)
        paired = []
        if len(rows) == 2 and not differing and not missing:
            def means(row):
                values = {}
                for episode in row["episodes"]:
                    utility = episode.get("U", episode.get("mean_satisfaction", episode.get("utility")))
                    if isinstance(utility, (int, float)):
                        values.setdefault(episode["scenario_id"], []).append(utility)
                return {key: statistics.mean(value) for key, value in values.items()}
            left, right = map(means, rows)
            paired = [{"scenario_id": key, "delta_U": right[key] - left[key]} for key in sorted(left.keys() & right.keys())]
        from ..research_stats import summarize_runs, compare_runs, _model_identity
        groups = {}
        for row in rows:
            group_key = _model_identity(row["manifest"])
            groups.setdefault(group_key, []).append(row)
        grouped = list(groups.values())
        def group_label(manifest):
            label=manifest.get('algorithm','未登记')
            paper_id=manifest.get('paper_baseline',{}).get('algorithm_id')
            if not manifest.get('model_not_applicable') and paper_id not in ('mlp_dqn','mlp_ppo'):
                label+=' · η='+str(manifest.get('config',{}).get('train',{}).get('target_entropy_ratio','未登记'))
            return label
        summaries = [{"algorithm": group_label(group[0]["manifest"]),
                      "experiment_id":hashlib.sha256(_model_identity(group[0]["manifest"]).encode()).hexdigest()[:16],
                      "summary": summarize_runs(group, phase=phase)} for group in grouped]
        comparison_stats = compare_runs(grouped[0], grouped[1], phase=phase) if len(grouped) == 2 else None
        statistics_compatible = all(item["summary"]["compatible"] for item in summaries)
        if comparison_stats is not None:
            statistics_compatible = statistics_compatible and comparison_stats["compatible"]
        for summary in [item["summary"] for item in summaries] + ([comparison_stats] if comparison_stats else []):
            differing = sorted(set(differing + summary.get("differing_fields", [])))
            missing = sorted(set(missing + summary.get("missing_fields", [])))
        if not statistics_compatible:
            paired = []
        return {"items": rows, "compatible": not differing and not missing and statistics_compatible, "differing_fields": differing,
                "missing_fields": missing, "paired_differences": paired, "ranking_available": False,
                "return_comparable": bool(return_compatible and not differing and not missing), "phase": phase,
                "statistics": {"groups": summaries, "comparison": comparison_stats},
                "seed_interval_available": bool(comparison_stats and comparison_stats["seed_interval_available"]),
                "note": "按算法与模型配置分组，同trial恢复分支去重；单seed不报告训练seed区间，场景区间不替代多seed证据"}

    @app.get("/api/v1/runs/{run_id}/coverage")
    def run_coverage(run_id: str):
        folder,manifest=resolve_run(run_id)
        item=manifest.get("coverage_snapshot",{}).get("artifacts",{}).get("audit")
        if not item:
            return {"status":"unavailable","reason":"该运行没有冻结训练域审计，不借用其他运行或当前数据审计"}
        path=(folder/item["path"]).resolve()
        if folder not in path.parents or not path.is_file():
            return {"status":"unavailable","reason":"冻结审计工件缺失"}
        if sha256(path)!=item["sha256"]:
            raise HTTPException(409,"Frozen coverage hash mismatch")
        return {"status":"recorded","source":"immutable_run_local","data":json.loads(path.read_text(encoding="utf-8")),
                "source_manifest_hash":manifest["coverage_snapshot"].get("source_manifest_hash")}

    @app.get("/api/v1/datasets/{dataset_id}/coverage")
    def coverage(dataset_id: str):
        if not re.fullmatch(r"[\w.\-]+", dataset_id) or dataset_id in (".", ".."):
            raise HTTPException(404, "Unknown dataset ID")
        candidates = [data_root / dataset_id / "训练场景覆盖审计.json", data_root / "训练场景覆盖审计.json",
                      data_root / dataset_id / "训练场景清单.json", data_root / "训练场景清单.json"]
        for path in candidates:
            resolved = path.resolve()
            if data_root not in resolved.parents or not resolved.is_file():
                continue
            data = json.loads(resolved.read_text(encoding="utf-8-sig"))
            if data.get("dataset_version", dataset_id) != dataset_id:
                continue
            return {"status": "recorded", "source": "dataset_manifest", "data": data}
        return {"status": "unavailable", "reason": "尚未登记训练域覆盖审计；CSV数量不代表覆盖率"}

    @app.get("/api/v1/artifacts/{artifact_id}")
    def artifact(artifact_id: str):
        for run_id, (folder, _) in discover().items():
            with connection(run_id) as conn:
                row = conn.execute("SELECT * FROM artifacts WHERE artifact_id=? AND status='available'", (artifact_id,)).fetchone()
            if row:
                path = (folder / row["path"]).resolve()
                if folder not in path.parents or not path.is_file():
                    raise HTTPException(410, "Artifact unavailable")
                if sha256(path) != row["sha256"]:
                    raise HTTPException(409, "Artifact hash mismatch")
                return FileResponse(path, filename=path.name)
        raise HTTPException(404, "Unknown artifact ID")

    if assets.is_dir():
        app.mount("/", StaticFiles(directory=assets, html=True), name="dashboard")
    return app
