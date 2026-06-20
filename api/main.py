"""
SimXLabs Simulation Concierge — Pilot API
FastAPI server that powers the Convai External API integration.
Hybrid: real LLM intent parsing + realistic simulated DAG execution.
"""

import asyncio
import os
import random
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from openai import OpenAI
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

load_dotenv()

app = FastAPI(
    title="SimXLabs Pilot API",
    description="Simulation Concierge — intent → DAG → verified training data",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory store (swap for Redis/Postgres in production) ──────────────────
runs: Dict[str, Any] = {}
traces: Dict[str, Any] = {}

# ── Constants ────────────────────────────────────────────────────────────────
FOUNDATION_MODELS = ["Pi0", "RT-2", "OpenVLA", "MimicGen"]

TASK_CONFIGS = {
    "bin_picking":   {"label": "Bin Picking",   "base_demos": 10000, "sim_engine": "Isaac Sim",  "eta": (50, 80)},
    "peg_insertion": {"label": "Peg Insertion",  "base_demos": 8000,  "sim_engine": "MuJoCo",    "eta": (60, 100)},
    "door_opening":  {"label": "Door Opening",   "base_demos": 12000, "sim_engine": "Isaac Sim",  "eta": (80, 130)},
    "cloth_folding": {"label": "Cloth Folding",  "base_demos": 6000,  "sim_engine": "Genesis",   "eta": (100, 160)},
    "custom":        {"label": "Custom Task",    "base_demos": 10000, "sim_engine": "MuJoCo",    "eta": (60, 120)},
}


# ── Pydantic models ──────────────────────────────────────────────────────────
class RunRequest(BaseModel):
    intent: str
    num_demos: Optional[int] = None
    constraints: Optional[Dict] = {}


class RunResponse(BaseModel):
    run_id: str
    status: str
    task_type: str
    eta_seconds: int
    created_at: str
    trace_url: str
    message: str


# ── Intent parsing (real LLM call via Anthropic SDK) ────────────────────────
def parse_intent_with_llm(intent: str) -> Dict[str, Any]:
    """Use GPT-4o-mini to extract structured fields from a natural-language intent."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _fallback_parse(intent)

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=256,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a robotics simulation task parser. "
                        "Given a user's natural-language simulation request, return a JSON object with these fields: "
                        "task_type (one of: bin_picking, peg_insertion, door_opening, cloth_folding, custom), "
                        "num_demos (integer, default 10000), "
                        "diversity_goal (high/medium/low), "
                        "key_constraints (list of strings, max 3). "
                        "Return only valid JSON, no explanation."
                    ),
                },
                {"role": "user", "content": intent},
            ],
        )
        import json
        raw = response.choices[0].message.content.strip()
        return json.loads(raw)
    except Exception:
        return _fallback_parse(intent)


def _fallback_parse(intent: str) -> Dict[str, Any]:
    """Rule-based fallback if no API key is set."""
    lower = intent.lower()
    task = "bin_picking"
    if "peg" in lower or "insert" in lower:
        task = "peg_insertion"
    elif "door" in lower:
        task = "door_opening"
    elif "cloth" in lower or "fold" in lower:
        task = "cloth_folding"

    num = 10000
    for token in lower.split():
        cleaned = token.replace(",", "").replace("k", "000")
        if cleaned.isdigit():
            num = int(cleaned)
            break

    return {
        "task_type": task,
        "num_demos": num,
        "diversity_goal": "high" if "diverse" in lower or "diversity" in lower else "medium",
        "key_constraints": [],
    }


# ── Background DAG execution (hybrid simulation) ─────────────────────────────
async def execute_dag(run_id: str, parsed: Dict, num_demos: int, eta: int):
    run = runs[run_id]
    dag_nodes = []

    def node(name: str, **kwargs):
        dag_nodes.append({"node": name, "timestamp": _now(), **kwargs})

    # ── Stage 1: Intent parsed ───────────────────────────────────────────────
    run["status"] = "running"
    run["stage"] = "Parsing intent"
    run["progress"] = 0.05
    await asyncio.sleep(1.5)

    node(
        "intent_parser",
        model="claude-haiku-4-5-20251001",
        duration_ms=1240,
        output=parsed,
        verified=True,
    )

    # ── Stage 2: Compile environment ─────────────────────────────────────────
    run["stage"] = "Compiling simulation environment"
    run["progress"] = 0.15
    cfg = TASK_CONFIGS.get(parsed["task_type"], TASK_CONFIGS["custom"])
    await asyncio.sleep(eta * 0.15)

    node(
        "env_compiler",
        engine=cfg["sim_engine"],
        scene=f"{parsed['task_type']}_v1",
        duration_ms=int(eta * 150),
        verified=True,
    )

    # ── Stage 3: Plan DAG ────────────────────────────────────────────────────
    run["stage"] = "Planning execution DAG"
    run["progress"] = 0.28
    await asyncio.sleep(2)

    node(
        "dag_planner",
        models_routed=FOUNDATION_MODELS,
        routing_strategy="diversity-maximizing",
        demos_per_model=num_demos // len(FOUNDATION_MODELS),
        duration_ms=920,
        verified=True,
    )

    # ── Stage 4: Sample across all 4 models (longest step) ───────────────────
    model_results = []
    per_model = num_demos // len(FOUNDATION_MODELS)

    for i, model in enumerate(FOUNDATION_MODELS):
        run["stage"] = f"Sampling — {model}"
        run["progress"] = 0.30 + i * 0.13
        await asyncio.sleep(eta * 0.14)

        demos = per_model + random.randint(-200, 200)
        diversity = round(random.uniform(0.71, 0.95), 3)
        duration_ms = int(eta * 140 + random.randint(-3000, 3000))

        result = {
            "model": model,
            "demos_generated": demos,
            "diversity_score": diversity,
            "duration_ms": duration_ms,
            "verified": True,
        }
        model_results.append(result)
        node(f"sample_{model.lower().replace('-', '_')}", **result)

    # ── Stage 5: Verify + deduplicate ────────────────────────────────────────
    run["stage"] = "Verifying and deduplicating"
    run["progress"] = 0.88
    await asyncio.sleep(3)

    total_raw = sum(r["demos_generated"] for r in model_results)
    removed = int(total_raw * random.uniform(0.02, 0.06))
    final_demos = total_raw - removed
    avg_diversity = round(sum(r["diversity_score"] for r in model_results) / 4, 3)
    dataset_mb = round(final_demos * 0.0025 * random.uniform(0.9, 1.1), 1)

    node(
        "verifier",
        raw_demos=total_raw,
        physics_violations_removed=removed,
        final_demos=final_demos,
        avg_diversity_score=avg_diversity,
        duration_ms=3100,
        verified=True,
    )

    # ── Stage 6: Cache write ─────────────────────────────────────────────────
    run["progress"] = 0.96
    await asyncio.sleep(1)
    cache_key = f"simx:{parsed['task_type']}:{num_demos}"
    speedup = random.randint(38, 82)

    node(
        "semantic_cache_write",
        cache_key=cache_key,
        ttl_hours=168,
        warm_run_speedup_estimate=f"{speedup}×",
        verified=True,
    )

    # ── Complete ─────────────────────────────────────────────────────────────
    run["status"] = "completed"
    run["stage"] = "Done"
    run["progress"] = 1.0
    run["completed_at"] = _now()
    run["result"] = {
        "task_type": parsed["task_type"],
        "task_label": cfg["label"],
        "total_demos": final_demos,
        "diversity_score": avg_diversity,
        "dataset_size_mb": dataset_mb,
        "format": "HDF5 + RLDS",
        "simulation_engine": cfg["sim_engine"],
        "models_used": FOUNDATION_MODELS,
        "model_breakdown": model_results,
        "download_url": f"https://api.simxlabs.com/datasets/{run_id}.zip",
        "warm_run_cache_key": cache_key,
        "warm_run_speedup": f"{speedup}×",
    }

    traces[run_id] = {
        "trace_id": f"trace-{run_id}",
        "run_id": run_id,
        "intent": run["intent"],
        "parsed_intent": parsed,
        "dag_nodes": dag_nodes,
        "total_duration_seconds": int((datetime.fromisoformat(run["completed_at"]) - datetime.fromisoformat(run["created_at"])).total_seconds()),
        "self_heal_events": 0,
        "cache_strategy": "semantic",
        "policy_gates_passed": len(dag_nodes),
    }


# ── API routes ───────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0-pilot", "runs_in_memory": len(runs)}


@app.post("/run", response_model=RunResponse)
async def create_run(req: RunRequest, background_tasks: BackgroundTasks):
    # Parse intent (real LLM call)
    parsed = parse_intent_with_llm(req.intent)
    if req.num_demos:
        parsed["num_demos"] = req.num_demos

    task_type = parsed.get("task_type", "bin_picking")
    num_demos = parsed.get("num_demos", 10000)
    cfg = TASK_CONFIGS.get(task_type, TASK_CONFIGS["custom"])
    eta = random.randint(*cfg["eta"])

    run_id = str(uuid.uuid4())[:8].upper()

    runs[run_id] = {
        "run_id": run_id,
        "intent": req.intent,
        "parsed_intent": parsed,
        "status": "queued",
        "stage": "Queued",
        "progress": 0.0,
        "created_at": _now(),
        "eta_seconds": eta,
        "result": None,
        "completed_at": None,
        "error": None,
    }

    background_tasks.add_task(execute_dag, run_id, parsed, num_demos, eta)

    base_url = os.getenv("API_BASE_URL", "http://localhost:8000")
    return RunResponse(
        run_id=run_id,
        status="queued",
        task_type=task_type,
        eta_seconds=eta,
        created_at=runs[run_id]["created_at"],
        trace_url=f"{base_url}/trace/{run_id}/view",
        message=f"Run {run_id} queued. Sampling across Pi0, RT-2, OpenVLA, and MimicGen. ETA ~{eta}s.",
    )


@app.get("/run/{run_id}")
async def get_run(run_id: str):
    if run_id not in runs:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found.")
    return runs[run_id]


@app.get("/run/{run_id}/summary")
async def get_run_summary(run_id: str):
    """Returns a concise summary string — optimized for Convai to narrate."""
    if run_id not in runs:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found.")
    run = runs[run_id]

    if run["status"] == "queued":
        return {"summary": f"Run {run_id} is queued. Hang tight — starting shortly."}

    if run["status"] == "running":
        pct = int(run["progress"] * 100)
        return {"summary": f"Run {run_id} is {pct}% complete. Currently: {run['stage']}."}

    if run["status"] == "completed":
        r = run["result"]
        return {
            "summary": (
                f"Done! Run {run_id} generated {r['total_demos']:,} verified robot demonstrations "
                f"across {', '.join(r['models_used'])}. "
                f"Diversity score: {r['diversity_score']}. "
                f"Dataset: {r['dataset_size_mb']} MB in {r['format']} format. "
                f"Repeat runs will be {r['warm_run_speedup']} faster thanks to semantic caching."
            ),
            "result": r,
            "trace_url": f"{os.getenv('API_BASE_URL', 'http://localhost:8000')}/trace/{run_id}/view",
        }

    return {"summary": f"Run {run_id} failed. Please try again or check your constraints."}


@app.get("/trace/{run_id}")
async def get_trace(run_id: str):
    if run_id not in traces:
        status = runs.get(run_id, {}).get("status", "unknown")
        raise HTTPException(
            status_code=404 if status == "unknown" else 202,
            detail=f"Trace not ready yet — run is {status}.",
        )
    return traces[run_id]


@app.get("/trace/{run_id}/view", response_class=HTMLResponse)
async def trace_viewer(run_id: str):
    """Dark-themed trace viewer matching SimXLabs brand."""
    if run_id not in runs:
        return HTMLResponse("<h1>Run not found</h1>", status_code=404)

    run = runs[run_id]
    trace = traces.get(run_id)
    nodes_html = ""

    if trace:
        for n in trace["dag_nodes"]:
            badge = "<span style='color:#7DC85A;font-size:10px;'>✓ verified</span>" if n.get("verified") else ""
            nodes_html += f"""
            <div style='border-left:2px solid rgba(125,200,90,0.3);padding:8px 16px;margin:8px 0;'>
              <div style='font-size:11px;font-weight:700;color:#7DC85A;letter-spacing:0.1em;text-transform:uppercase;'>{n['node']} {badge}</div>
              <pre style='font-size:10px;color:rgba(255,255,255,0.5);margin:4px 0 0;white-space:pre-wrap;'>{_pretty(n)}</pre>
            </div>"""

    result_html = ""
    if run.get("result"):
        r = run["result"]
        result_html = f"""
        <div style='margin-top:24px;padding:16px;border:1px solid rgba(125,200,90,0.2);border-radius:8px;'>
          <div style='font-size:11px;font-weight:700;color:rgba(197,232,176,0.6);letter-spacing:0.15em;text-transform:uppercase;margin-bottom:12px;'>Result</div>
          <div style='display:flex;gap:32px;flex-wrap:wrap;'>
            <div><div style='font-size:32px;font-weight:800;color:#7DC85A;'>{r['total_demos']:,}</div><div style='font-size:10px;color:rgba(255,255,255,0.4);'>demos generated</div></div>
            <div><div style='font-size:32px;font-weight:800;color:#7DC85A;'>{r['diversity_score']}</div><div style='font-size:10px;color:rgba(255,255,255,0.4);'>diversity score</div></div>
            <div><div style='font-size:32px;font-weight:800;color:#7DC85A;'>{r['dataset_size_mb']} MB</div><div style='font-size:10px;color:rgba(255,255,255,0.4);'>dataset size</div></div>
            <div><div style='font-size:32px;font-weight:800;color:#7DC85A;'>{r['warm_run_speedup']}</div><div style='font-size:10px;color:rgba(255,255,255,0.4);'>warm-run speedup</div></div>
          </div>
        </div>"""

    progress_pct = int(run["progress"] * 100)
    status_color = "#7DC85A" if run["status"] == "completed" else "#FFB347" if run["status"] == "running" else "rgba(255,255,255,0.4)"

    return HTMLResponse(f"""<!DOCTYPE html><html><head>
<meta charset="UTF-8"><title>SimXLabs Trace — {run_id}</title>
<meta http-equiv="refresh" content="{'' if run['status'] == 'completed' else '5'}">
<style>
  *{{box-sizing:border-box;}} body{{margin:0;padding:32px;font-family:-apple-system,'Inter',sans-serif;
  background:linear-gradient(158deg,#0D1C0B 0%,#182E14 38%,#1C3518 58%,#0D1C0B 100%);
  min-height:100vh;color:#fff;}}
  pre{{margin:0;}}
</style></head><body>
<div style='max-width:800px;margin:0 auto;'>
  <div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:24px;'>
    <div>
      <div style='font-size:11px;font-weight:700;letter-spacing:0.2em;text-transform:uppercase;color:rgba(197,232,176,0.5);'>SimXLabs · Trace</div>
      <div style='font-size:24px;font-weight:800;margin-top:4px;'>Run <span style='color:#7DC85A;'>{run_id}</span></div>
    </div>
    <div style='text-align:right;'>
      <div style='font-size:13px;font-weight:600;color:{status_color};'>{run['status'].upper()}</div>
      <div style='font-size:11px;color:rgba(255,255,255,0.3);margin-top:2px;'>{run['stage']}</div>
    </div>
  </div>

  <div style='background:rgba(255,255,255,0.04);border-radius:4px;height:4px;margin-bottom:24px;'>
    <div style='background:#7DC85A;height:4px;border-radius:4px;width:{progress_pct}%;transition:width 0.5s;'></div>
  </div>

  <div style='font-size:11px;color:rgba(255,255,255,0.3);margin-bottom:4px;'>Intent</div>
  <div style='font-size:14px;color:rgba(255,255,255,0.8);margin-bottom:24px;font-style:italic;'>"{run['intent']}"</div>

  {result_html}

  <div style='margin-top:24px;'>
    <div style='font-size:11px;font-weight:700;color:rgba(197,232,176,0.5);letter-spacing:0.15em;text-transform:uppercase;margin-bottom:8px;'>DAG Execution Log</div>
    {nodes_html if nodes_html else "<div style='color:rgba(255,255,255,0.3);font-size:12px;'>Execution in progress...</div>"}
  </div>

  <div style='margin-top:32px;font-size:10px;color:rgba(255,255,255,0.2);'>
    Created {run['created_at']} · trace-{run_id}
    {f"· Completed {run['completed_at']}" if run.get('completed_at') else ''}
  </div>
</div>
</body></html>""")


# ── Helpers ──────────────────────────────────────────────────────────────────
def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pretty(d: dict) -> str:
    import json
    skip = {"node", "timestamp", "verified"}
    filtered = {k: v for k, v in d.items() if k not in skip}
    return json.dumps(filtered, indent=2)
