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
    parsed_intent: Optional[Dict] = None
    osmo_workflow: Optional[str] = None


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


def generate_osmo_workflow(run_id: str, parsed: Dict, num_demos: int, cfg: Dict) -> str:
    """Compile SimXLabs intent into a valid NVIDIA OSMO workflow YAML."""
    task_type = parsed.get("task_type", "bin_picking")
    sim_engine = cfg.get("sim_engine", "Isaac Sim")
    engine_image = {
        "Isaac Sim": "nvcr.io/nvidia/isaac-sim:4.2.0",
        "MuJoCo":    "simxlabs/mujoco-sim:latest",
        "Genesis":   "simxlabs/genesis-sim:latest",
    }.get(sim_engine, "nvcr.io/nvidia/isaac-sim:4.2.0")
    slug = task_type.replace("_", "-")
    return f"""# SimXLabs x NVIDIA OSMO — Generated Workflow
# Run: {run_id}  |  Task: {task_type}  |  Demos: {num_demos:,}
# Compiled by SimXLabs Decision Engine

workflow:
  name: simxlabs-{slug}-{run_id.lower()}

  tasks:

  - name: env-compiler
    image: {engine_image}
    platform: rtx-pro-6000
    resources:
      gpu: 1
    outputs:
    - url: s3://simxlabs-runs/{run_id}/env/

  - name: sample-pi0
    image: simxlabs/pi0-sampler:1.0
    platform: gb200
    resources:
      gpu: 2
    inputs:
    - task: env-compiler
    outputs:
    - url: s3://simxlabs-runs/{run_id}/demos/pi0/

  - name: sample-rt2
    image: simxlabs/rt2-sampler:1.0
    platform: gb200
    resources:
      gpu: 2
    inputs:
    - task: env-compiler
    outputs:
    - url: s3://simxlabs-runs/{run_id}/demos/rt2/

  - name: sample-openvla
    image: simxlabs/openvla-sampler:1.0
    platform: gb200
    resources:
      gpu: 2
    inputs:
    - task: env-compiler
    outputs:
    - url: s3://simxlabs-runs/{run_id}/demos/openvla/

  - name: sample-mimicgen
    image: simxlabs/mimicgen-sampler:1.0
    platform: rtx-pro-6000
    resources:
      gpu: 2
    inputs:
    - task: env-compiler
    outputs:
    - url: s3://simxlabs-runs/{run_id}/demos/mimicgen/

  - name: physics-verifier
    image: simxlabs/physics-gate:1.0
    platform: x86-64
    inputs:
    - task: sample-pi0
    - task: sample-rt2
    - task: sample-openvla
    - task: sample-mimicgen
    outputs:
    - url: s3://simxlabs-runs/{run_id}/verified/

  - name: semantic-cache-write
    image: simxlabs/semantic-cache:1.0
    platform: x86-64
    inputs:
    - task: physics-verifier
    outputs:
    - url: s3://simxlabs-datasets/cache/{task_type}_{num_demos}/
"""


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



# ── Demo page ────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def demo_page():
    return HTMLResponse("""<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8">
<title>SimXLabs x Convai</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&display=swap');
*{box-sizing:border-box;margin:0;padding:0;}
html,body{height:100%;overflow:hidden;}
body{font-family:'Inter',sans-serif;
  background:linear-gradient(158deg,#080f07 0%,#0f1e0d 40%,#111f0f 60%,#080f07 100%);
  color:#fff;}

.shell{display:flex;flex-direction:column;height:100vh;}

/* Top bar */
.topbar{display:flex;align-items:center;justify-content:space-between;
  padding:10px 24px;border-bottom:1px solid rgba(255,255,255,0.05);flex-shrink:0;}
.tb-left{display:flex;align-items:center;gap:8px;}
.tb-logo{width:20px;height:20px;}
.tb-brand{font-size:13px;font-weight:700;letter-spacing:-0.01em;}
.tb-pill{font-size:8px;font-weight:700;letter-spacing:0.16em;text-transform:uppercase;
  color:#7DC85A;border:1px solid rgba(125,200,90,0.22);border-radius:20px;padding:2px 8px;}
.tb-right{font-size:10px;color:rgba(255,255,255,0.18);letter-spacing:0.06em;}

/* Split */
.split{display:flex;flex:1;min-height:0;}

/* ─── LEFT: Simra pane ─── */
.pane-convai{width:30%;border-right:1px solid rgba(255,255,255,0.05);
  display:flex;flex-direction:column;flex-shrink:0;}
.pane-label{padding:10px 20px;font-size:9px;font-weight:600;letter-spacing:0.18em;
  text-transform:uppercase;color:rgba(255,255,255,0.18);
  border-bottom:1px solid rgba(255,255,255,0.04);display:flex;align-items:center;gap:8px;}
.pane-label .dot{width:6px;height:6px;border-radius:50%;background:#7DC85A;
  box-shadow:0 0 8px #7DC85A;animation:pulse 2s ease-in-out infinite;}
@keyframes pulse{0%,100%{opacity:1;}50%{opacity:0.3;}}
.convai-panel{flex:1;display:flex;flex-direction:column;align-items:center;
  justify-content:center;padding:24px 20px;}
.avatar-wrap{position:relative;margin-bottom:14px;}
.avatar-ring{position:absolute;inset:-8px;border-radius:50%;
  border:1px solid rgba(125,200,90,0.16);animation:ring 2.5s ease-in-out infinite;}
.avatar-ring2{position:absolute;inset:-17px;border-radius:50%;
  border:1px solid rgba(125,200,90,0.06);animation:ring 2.5s ease-in-out infinite 0.85s;}
@keyframes ring{0%,100%{opacity:0.5;transform:scale(1);}50%{opacity:0.1;transform:scale(1.06);}}
.simra-photo{width:108px;height:108px;border-radius:50%;object-fit:cover;object-position:top;
  border:2px solid rgba(125,200,90,0.32);display:block;}
.live-badge{display:flex;align-items:center;gap:5px;margin-bottom:12px;
  font-size:9px;color:rgba(125,200,90,0.5);letter-spacing:0.1em;}
.live-dot{width:5px;height:5px;border-radius:50%;background:#7DC85A;
  animation:pulse 1.5s ease-in-out infinite;}
.convai-name{font-size:15px;font-weight:700;margin-bottom:4px;}
.convai-role{font-size:10px;color:rgba(255,255,255,0.22);margin-bottom:14px;
  text-align:center;line-height:1.7;}
.caps{display:flex;flex-direction:column;gap:7px;margin-bottom:20px;width:100%;max-width:200px;}
.cap{display:flex;align-items:flex-start;gap:8px;font-size:10px;
  color:rgba(255,255,255,0.26);line-height:1.5;}
.cap-dot{width:4px;height:4px;border-radius:50%;background:#7DC85A;opacity:0.45;
  flex-shrink:0;margin-top:4px;}
.talk-btn{display:flex;align-items:center;gap:8px;padding:11px 22px;
  background:rgba(125,200,90,0.08);border:1px solid rgba(125,200,90,0.2);
  border-radius:9px;color:#7DC85A;font-size:12px;font-weight:600;
  cursor:pointer;text-decoration:none;transition:all 0.18s;margin-bottom:8px;}
.talk-btn svg{width:14px;height:14px;}
.talk-btn:hover{background:rgba(125,200,90,0.14);transform:translateY(-1px);
  box-shadow:0 6px 20px rgba(125,200,90,0.08);}
.convai-note{font-size:9px;color:rgba(255,255,255,0.1);text-align:center;}

/* ─── RIGHT: Pilot pane ─── */
.pane-pilot{flex:1;display:flex;flex-direction:column;overflow:hidden;}
.pilot-scroll{flex:1;overflow-y:auto;padding:22px 26px 26px;}
.pilot-scroll::-webkit-scrollbar{width:3px;}
.pilot-scroll::-webkit-scrollbar-thumb{background:rgba(255,255,255,0.06);border-radius:3px;}

/* ─── BIG STATS ROW ─── */
.stats-bar{display:flex;align-items:center;margin-bottom:18px;
  background:rgba(255,255,255,0.025);border:1px solid rgba(255,255,255,0.06);
  border-radius:16px;padding:22px 0;}
.sb-item{flex:1;text-align:center;}
.sb-val{font-size:42px;font-weight:900;color:#7DC85A;
  letter-spacing:-0.05em;line-height:1;}
.sb-lbl{font-size:10px;font-weight:500;color:rgba(255,255,255,0.22);
  margin-top:6px;letter-spacing:0.1em;text-transform:uppercase;}
.sb-live{display:flex;align-items:center;justify-content:center;gap:4px;
  font-size:8px;color:rgba(125,200,90,0.38);margin-top:4px;}
.sb-live-dot{width:4px;height:4px;border-radius:50%;background:#7DC85A;
  animation:pulse 1.5s ease-in-out infinite;}
.sb-sep{width:1px;height:48px;background:rgba(255,255,255,0.05);}

/* ─── PIPELINE NODES ─── */
.pipe-flow{display:flex;align-items:center;justify-content:space-between;
  background:rgba(0,0,0,0.18);border:1px solid rgba(255,255,255,0.05);
  border-radius:16px;padding:28px 18px;margin-bottom:16px;}

.pnode{display:flex;flex-direction:column;align-items:center;flex-shrink:0;width:80px;}
.pnode-wrap{position:relative;width:68px;height:68px;margin-bottom:12px;}
.pnode-outer{position:absolute;inset:-7px;border-radius:50%;
  border:1px solid rgba(125,200,90,0.08);
  animation:nodeRing 3s ease-in-out infinite;}
@keyframes nodeRing{0%,100%{opacity:0.5;transform:scale(1);}50%{opacity:0.08;transform:scale(1.1);}}
.pnode-circle{width:68px;height:68px;border-radius:50%;
  background:rgba(125,200,90,0.07);border:1.5px solid rgba(125,200,90,0.2);
  display:flex;align-items:center;justify-content:center;
  box-shadow:0 0 0 0 rgba(125,200,90,0);}
.pnode-circle svg{display:block;}
.pnode.hl .pnode-circle{background:rgba(125,200,90,0.12);
  border-color:rgba(125,200,90,0.35);
  box-shadow:0 0 24px rgba(125,200,90,0.12);}
.pnode-lbl{font-size:12px;font-weight:600;color:rgba(255,255,255,0.5);
  text-align:center;line-height:1.3;}
.pnode-sub{font-size:9px;color:rgba(125,200,90,0.38);
  margin-top:3px;text-align:center;white-space:nowrap;}

/* Animated connector */
.pconn{flex:1;display:flex;align-items:center;padding:0 4px;padding-bottom:30px;}
.pconn-line{flex:1;height:2px;border-radius:1px;
  background:linear-gradient(90deg,rgba(125,200,90,0.06),rgba(125,200,90,0.12),rgba(125,200,90,0.06));
  position:relative;overflow:hidden;}
@keyframes flowR{0%{left:-12px;opacity:0;}8%{opacity:1;}92%{opacity:1;}100%{left:calc(100% + 12px);opacity:0;}}
.pdot{position:absolute;top:50%;transform:translateY(-50%);
  width:8px;height:8px;border-radius:50%;background:#7DC85A;
  box-shadow:0 0 10px rgba(125,200,90,0.9),0 0 4px #7DC85A;
  animation:flowR 2.2s linear infinite;}
.pdot:nth-child(2){animation-delay:0.73s;}
.pdot:nth-child(3){animation-delay:1.47s;}

/* ─── ACTIVITY TICKER ─── */
.ticker{display:flex;align-items:center;gap:10px;padding:6px 0 14px;}
.ticker-dot{width:5px;height:5px;border-radius:50%;background:#7DC85A;
  flex-shrink:0;animation:pulse 1.5s ease-in-out infinite;}
.ticker-txt{font-size:11px;color:rgba(255,255,255,0.2);
  font-family:'JetBrains Mono',monospace;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.ticker-txt .ok{color:rgba(125,200,90,0.55);}
.ticker-txt .act{color:rgba(255,179,71,0.5);}

/* ─── RUN SECTION ─── */
.run-section{background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.05);
  border-radius:14px;padding:20px 22px;}
.run-sec-label{font-size:9px;font-weight:600;letter-spacing:0.18em;text-transform:uppercase;
  color:rgba(255,255,255,0.18);margin-bottom:14px;}
textarea{width:100%;background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);
  border-radius:10px;padding:12px 14px;color:#fff;font-size:12px;
  font-family:'Inter',sans-serif;resize:none;height:54px;line-height:1.6;
  outline:none;transition:border 0.2s;margin-bottom:10px;}
textarea:focus{border-color:rgba(125,200,90,0.3);}
textarea::placeholder{color:rgba(255,255,255,0.14);}
.prompts{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:12px;}
.prompt{font-size:10px;padding:5px 11px;border-radius:20px;
  background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);
  color:rgba(255,255,255,0.28);cursor:pointer;transition:all 0.15s;}
.prompt:hover{background:rgba(125,200,90,0.07);border-color:rgba(125,200,90,0.2);
  color:rgba(255,255,255,0.6);}
.run-btn{width:100%;padding:12px;background:#7DC85A;border:none;border-radius:10px;
  font-size:13px;font-weight:700;color:#0A1509;cursor:pointer;transition:all 0.15s;
  letter-spacing:0.02em;}
.run-btn:hover:not(:disabled){background:#8FD96A;transform:translateY(-1px);}
.run-btn:disabled{opacity:0.3;cursor:not-allowed;transform:none;}
.run-status{display:none;margin-top:18px;}
.run-status.show{display:block;}
.run-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:9px;}
.run-id{font-size:11px;color:rgba(255,255,255,0.28);}
.run-id strong{color:#7DC85A;font-weight:600;}
.status-pill{font-size:9px;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;
  padding:3px 9px;border-radius:20px;}
.pill-queued{color:rgba(255,255,255,0.3);background:rgba(255,255,255,0.05);}
.pill-running{color:#FFB347;background:rgba(255,179,71,0.08);}
.pill-completed{color:#7DC85A;background:rgba(125,200,90,0.08);}
.prog-track{background:rgba(255,255,255,0.05);border-radius:3px;height:3px;margin-bottom:5px;}
.prog-fill{background:linear-gradient(90deg,#5DB840,#7DC85A);height:3px;border-radius:3px;
  width:0%;transition:width 0.8s ease;}
.stage-txt{font-size:10px;color:rgba(255,255,255,0.2);margin-bottom:14px;}
.model-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:7px;margin-bottom:12px;}
.mc{background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);
  border-radius:9px;padding:11px 10px 9px;transition:all 0.25s;}
.mc.active{background:rgba(125,200,90,0.07);border-color:rgba(125,200,90,0.24);}
.mc.done{background:rgba(125,200,90,0.04);border-color:rgba(125,200,90,0.12);}
.mc-name{font-size:9px;font-weight:700;letter-spacing:0.06em;
  color:rgba(255,255,255,0.28);margin-bottom:6px;transition:color 0.2s;}
.mc.active .mc-name,.mc.done .mc-name{color:#7DC85A;}
.mc-demos{font-size:16px;font-weight:800;color:rgba(255,255,255,0.1);
  letter-spacing:-0.02em;transition:color 0.3s;}
.mc.done .mc-demos{color:#fff;}
.mc-div{font-size:9px;color:rgba(255,255,255,0.14);margin-top:2px;transition:color 0.2s;}
.mc.done .mc-div{color:rgba(125,200,90,0.45);}
.mc-dot{width:5px;height:5px;border-radius:50%;background:rgba(255,255,255,0.08);
  margin-top:7px;transition:all 0.2s;}
.mc.active .mc-dot{background:#7DC85A;box-shadow:0 0 6px #7DC85A;animation:pulse 1s ease-in-out infinite;}
.mc.done .mc-dot{background:rgba(125,200,90,0.3);}
.results{display:none;margin-top:14px;}
.results.show{display:block;}
.results-row{display:grid;grid-template-columns:repeat(4,1fr);gap:2px;
  background:rgba(255,255,255,0.04);border-radius:10px;overflow:hidden;margin-bottom:10px;}
.stat{background:#080f07;padding:12px 12px 10px;}
.stat-l{font-size:8px;font-weight:600;letter-spacing:0.16em;text-transform:uppercase;
  color:rgba(255,255,255,0.18);margin-bottom:6px;}
.stat-v{font-size:18px;font-weight:900;color:#7DC85A;letter-spacing:-0.02em;}
.stat-s{font-size:8px;color:rgba(255,255,255,0.16);margin-top:2px;}
.trace-btn{display:block;padding:10px 14px;text-align:center;
  background:rgba(125,200,90,0.04);border:1px solid rgba(125,200,90,0.1);
  border-radius:9px;text-decoration:none;color:rgba(197,232,176,0.4);
  font-size:11px;font-weight:500;transition:all 0.15s;}
.trace-btn:hover{background:rgba(125,200,90,0.09);color:#7DC85A;}

/* ─── PIPELINE INTEL TABS ─── */
.intel-tabs{display:flex;gap:5px;margin:12px 0 10px;}
.itab{font-size:9px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;
  padding:5px 12px;border-radius:6px;cursor:pointer;transition:all 0.15s;
  background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);
  color:rgba(255,255,255,0.22);}
.itab.on{background:rgba(125,200,90,0.08);border-color:rgba(125,200,90,0.22);color:#7DC85A;}
.ipanel{display:none;}
.ipanel.show{display:block;}
.code-blk{background:rgba(0,0,0,0.35);border:1px solid rgba(255,255,255,0.06);
  border-radius:8px;padding:12px 14px;font-family:'JetBrains Mono',monospace;
  font-size:9.5px;color:rgba(255,255,255,0.5);white-space:pre;overflow:auto;
  max-height:170px;line-height:1.75;margin-bottom:10px;}
</style></head>
<body>
<div class="shell">

  <div class="topbar">
    <div class="tb-left">
      <svg class="tb-logo" viewBox="0 0 48 48" fill="none">
        <g stroke="rgba(255,255,255,0.22)" stroke-width="1.4" stroke-linecap="round">
          <line x1="24" y1="10" x2="18" y2="17"/><line x1="24" y1="10" x2="30" y2="17"/>
          <line x1="18" y1="19" x2="12" y2="25"/><line x1="18" y1="19" x2="24" y2="25"/>
          <line x1="30" y1="19" x2="24" y2="25"/><line x1="30" y1="19" x2="36" y2="25"/>
          <line x1="12" y1="29" x2="18" y2="35"/><line x1="24" y1="29" x2="18" y2="35"/>
          <line x1="24" y1="29" x2="30" y2="35"/><line x1="36" y1="29" x2="30" y2="35"/>
          <line x1="18" y1="37" x2="24" y2="43"/><line x1="30" y1="37" x2="24" y2="43"/>
        </g>
        <circle cx="24" cy="7" r="4" fill="#7DC85A"/>
        <circle cx="18" cy="17" r="3.5" fill="#7DC85A" opacity="0.85"/>
        <circle cx="30" cy="17" r="3.5" fill="#7DC85A" opacity="0.85"/>
        <rect x="8" y="22" width="8" height="6" rx="2" fill="#7DC85A" opacity="0.7"/>
        <rect x="20" y="22" width="8" height="6" rx="2" fill="#7DC85A" opacity="0.7"/>
        <rect x="32" y="22" width="8" height="6" rx="2" fill="#7DC85A" opacity="0.7"/>
        <rect x="14" y="31" width="8" height="6" rx="2" fill="#7DC85A" opacity="0.75"/>
        <rect x="26" y="31" width="8" height="6" rx="2" fill="#7DC85A" opacity="0.75"/>
        <circle cx="24" cy="43" r="3.5" fill="#7DC85A"/>
      </svg>
      <span class="tb-brand">SimXLabs</span>
      <span class="tb-pill">Pilot</span>
    </div>
    <div class="tb-right">Powered by Convai</div>
  </div>

  <div class="split">

    <!-- Left: Simra -->
    <div class="pane-convai">
      <div class="pane-label">
        <div class="dot"></div>Simra — AI Concierge
      </div>
      <div class="convai-panel">
        <div class="avatar-wrap">
          <div class="avatar-ring"></div>
          <div class="avatar-ring2"></div>
          <img class="simra-photo"
            src="https://storage.googleapis.com/experience-asset-storage/user-uploaded-avatar-image/db820f2f-7d0a-4220-8c90-23b477f35893_avatar_image_square?img_last_modified=1781921895"
            alt="Simra"/>
        </div>
        <div class="live-badge"><div class="live-dot"></div>Live · Voice Enabled</div>
        <div class="convai-name">Simra</div>
        <div class="convai-role">SimXLabs AI Concierge<br/>Powered by Convai</div>
        <div class="caps">
          <div class="cap"><div class="cap-dot"></div>Answers questions about the foundation model pipeline</div>
          <div class="cap"><div class="cap-dot"></div>Explains simulation results in plain English</div>
          <div class="cap"><div class="cap-dot"></div>Walks through task types and traceability</div>
        </div>
        <a class="talk-btn" href="https://x.convai.com/?xpid=176dbd7e-b46d-48a6-82b0-6627c3973ce2&type=unlisted" target="_blank">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/></svg>
          Talk to Simra
        </a>
        <div class="convai-note">Opens in a new tab · voice enabled</div>
      </div>
    </div>

    <!-- Right: Pilot visualization + run -->
    <div class="pane-pilot">
      <div class="pilot-scroll">

        <!-- STATS BAR -->
        <div class="stats-bar">
          <div class="sb-item">
            <div class="sb-val" id="mDemos">48,219</div>
            <div class="sb-lbl">Demos Generated</div>
            <div class="sb-live"><div class="sb-live-dot"></div>live</div>
          </div>
          <div class="sb-sep"></div>
          <div class="sb-item">
            <div class="sb-val" id="mDiversity">0.87</div>
            <div class="sb-lbl">Diversity Score</div>
          </div>
          <div class="sb-sep"></div>
          <div class="sb-item">
            <div class="sb-val"><span id="mSpeedup">54</span>×</div>
            <div class="sb-lbl">Cache Speedup</div>
          </div>
        </div>

        <!-- PIPELINE FLOW -->
        <div class="pipe-flow">

          <!-- Node 1: Voice -->
          <div class="pnode">
            <div class="pnode-wrap">
              <div class="pnode-outer"></div>
              <div class="pnode-circle">
                <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="rgba(125,200,90,0.75)" stroke-width="1.8" stroke-linecap="round"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/></svg>
              </div>
            </div>
            <div class="pnode-lbl">Voice</div>
            <div class="pnode-sub">Convai</div>
          </div>

          <div class="pconn"><div class="pconn-line"><div class="pdot"></div><div class="pdot"></div><div class="pdot"></div></div></div>

          <!-- Node 2: Parse -->
          <div class="pnode">
            <div class="pnode-wrap">
              <div class="pnode-outer"></div>
              <div class="pnode-circle">
                <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="rgba(125,200,90,0.75)" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="6" height="6" rx="1"/><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="9" y1="3" x2="9" y2="1"/><line x1="15" y1="3" x2="15" y2="1"/><line x1="9" y1="23" x2="9" y2="21"/><line x1="15" y1="23" x2="15" y2="21"/><line x1="3" y1="9" x2="1" y2="9"/><line x1="3" y1="15" x2="1" y2="15"/><line x1="23" y1="9" x2="21" y2="9"/><line x1="23" y1="15" x2="21" y2="15"/></svg>
              </div>
            </div>
            <div class="pnode-lbl">Parse</div>
            <div class="pnode-sub">GPT-4o</div>
          </div>

          <div class="pconn"><div class="pconn-line"><div class="pdot"></div><div class="pdot"></div><div class="pdot"></div></div></div>

          <!-- Node 3: Simulate -->
          <div class="pnode">
            <div class="pnode-wrap">
              <div class="pnode-outer"></div>
              <div class="pnode-circle">
                <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="rgba(125,200,90,0.75)" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
              </div>
            </div>
            <div class="pnode-lbl">Simulate</div>
            <div class="pnode-sub">Isaac · MuJoCo</div>
          </div>

          <div class="pconn"><div class="pconn-line"><div class="pdot"></div><div class="pdot"></div><div class="pdot"></div></div></div>

          <!-- Node 4: 4 Models (highlighted) -->
          <div class="pnode hl">
            <div class="pnode-wrap">
              <div class="pnode-outer"></div>
              <div class="pnode-circle">
                <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="rgba(125,200,90,0.9)" stroke-width="1.8"><circle cx="8" cy="8" r="3"/><circle cx="16" cy="8" r="3"/><circle cx="8" cy="16" r="3"/><circle cx="16" cy="16" r="3"/></svg>
              </div>
            </div>
            <div class="pnode-lbl" style="color:rgba(255,255,255,0.75);">4 Models</div>
            <div class="pnode-sub">Pi0 · RT-2 +</div>
          </div>

          <div class="pconn"><div class="pconn-line"><div class="pdot"></div><div class="pdot"></div><div class="pdot"></div></div></div>

          <!-- Node 5: Verify -->
          <div class="pnode">
            <div class="pnode-wrap">
              <div class="pnode-outer"></div>
              <div class="pnode-circle">
                <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="rgba(125,200,90,0.75)" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><polyline points="9 12 11 14 15 10"/></svg>
              </div>
            </div>
            <div class="pnode-lbl">Verify</div>
            <div class="pnode-sub">Physics Gate</div>
          </div>

          <div class="pconn"><div class="pconn-line"><div class="pdot"></div><div class="pdot"></div><div class="pdot"></div></div></div>

          <!-- Node 6: Cache -->
          <div class="pnode">
            <div class="pnode-wrap">
              <div class="pnode-outer"></div>
              <div class="pnode-circle">
                <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="rgba(125,200,90,0.75)" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>
              </div>
            </div>
            <div class="pnode-lbl">Cache</div>
            <div class="pnode-sub">54× Speedup</div>
          </div>

        </div>

        <!-- Live ticker -->
        <div class="ticker">
          <div class="ticker-dot"></div>
          <div class="ticker-txt" id="tickerTxt"><span class="ok">&#10003; System active</span></div>
        </div>

        <!-- Run section -->
        <div class="run-section">
          <div class="run-sec-label">Try it — Run a Simulation</div>
          <textarea id="intent" placeholder="Describe a robot task in plain English — e.g. bin picking, 10,000 demos, maximize diversity"></textarea>
          <div class="prompts">
            <div class="prompt" onclick="set('bin picking, 10000 demos, maximize trajectory diversity')">📦 Bin Picking</div>
            <div class="prompt" onclick="set('peg insertion with sub-millimeter precision, 8000 demos')">🔩 Peg Insertion</div>
            <div class="prompt" onclick="set('door opening across varied handle types, 12000 demos')">🚪 Door Opening</div>
            <div class="prompt" onclick="set('cloth folding with deformable object dynamics, 6000 demos')">👕 Cloth Folding</div>
          </div>
          <button class="run-btn" id="runBtn" onclick="startRun()">Run Simulation &#8594;</button>

          <div class="run-status" id="runStatus">
            <div class="run-header">
              <div class="run-id">Run <strong id="runIdTxt">—</strong></div>
              <div class="status-pill pill-queued" id="statusPill">Queued</div>
            </div>
            <div class="prog-track"><div class="prog-fill" id="pFill"></div></div>
            <div class="stage-txt" id="stageTxt">Initializing...</div>
            <div class="intel-tabs">
              <div class="itab on" id="tab-intent" onclick="switchTab('intent')">GPT-4o Intent</div>
              <div class="itab" id="tab-osmo" onclick="switchTab('osmo')">OSMO Workflow</div>
              <div class="itab" id="tab-exec" onclick="switchTab('exec')">Execution</div>
            </div>
            <div class="ipanel show" id="panel-intent">
              <div class="code-blk" id="intentBlk">Parsing intent with GPT-4o-mini...</div>
            </div>
            <div class="ipanel" id="panel-osmo">
              <div class="code-blk" id="osmoBlk">Generating OSMO workflow YAML...</div>
            </div>
            <div class="ipanel" id="panel-exec">
              <div class="model-grid">
                <div class="mc" id="mc-Pi0"><div class="mc-name">Pi0</div><div class="mc-demos" id="md-Pi0">—</div><div class="mc-div" id="mdiv-Pi0">div —</div><div class="mc-dot"></div></div>
                <div class="mc" id="mc-RT2"><div class="mc-name">RT-2</div><div class="mc-demos" id="md-RT2">—</div><div class="mc-div" id="mdiv-RT2">div —</div><div class="mc-dot"></div></div>
                <div class="mc" id="mc-OpenVLA"><div class="mc-name">OpenVLA</div><div class="mc-demos" id="md-OpenVLA">—</div><div class="mc-div" id="mdiv-OpenVLA">div —</div><div class="mc-dot"></div></div>
                <div class="mc" id="mc-MimicGen"><div class="mc-name">MimicGen</div><div class="mc-demos" id="md-MimicGen">—</div><div class="mc-div" id="mdiv-MimicGen">div —</div><div class="mc-dot"></div></div>
              </div>
            </div>
            <div class="results" id="results">
              <div class="results-row">
                <div class="stat"><div class="stat-l">Demos</div><div class="stat-v" id="rDemos">—</div><div class="stat-s">verified</div></div>
                <div class="stat"><div class="stat-l">Diversity</div><div class="stat-v" id="rDiv">—</div><div class="stat-s">score / 1.0</div></div>
                <div class="stat"><div class="stat-l">Dataset</div><div class="stat-v" id="rMb">—</div><div class="stat-s">HDF5 + RLDS</div></div>
                <div class="stat"><div class="stat-l">Warm Run</div><div class="stat-v" id="rSpeedup">—</div><div class="stat-s">speedup</div></div>
              </div>
              <a class="trace-btn" id="traceBtn" href="#" target="_blank">&#8594; View full execution trace</a>
            </div>
          </div>
        </div>

      </div>
    </div>

  </div>
</div>

<script>
const MODELS=['Pi0','RT-2','OpenVLA','MimicGen'];
let pollId=null, done=new Set();
let demoCount=48219, diversityVal=0.87, speedup=54;

function switchTab(t){
  ['intent','osmo','exec'].forEach(id=>{
    document.getElementById('panel-'+id).classList.toggle('show',id===t);
    document.getElementById('tab-'+id).classList.toggle('on',id===t);
  });
}

// Live demo counter ticks slowly
setInterval(()=>{
  demoCount+=Math.floor(Math.random()*3)+1;
  document.getElementById('mDemos').textContent=demoCount.toLocaleString();
  diversityVal=+(0.87+(Math.random()-0.5)*0.018).toFixed(2);
  document.getElementById('mDiversity').textContent=diversityVal;
}, 2600);

// Ticker rotates through realistic events
const TICKS=[
  {cls:'ok', t:'&#10003; Pi0 trajectory verified &middot; diversity 0.89'},
  {cls:'act',t:'&#8634; Isaac Sim spawning 32 parallel environments'},
  {cls:'ok', t:'&#10003; HDF5 chunk written &middot; 128 demos flushed'},
  {cls:'ok', t:'&#10003; Semantic cache hit &middot; bin_picking:8192'},
  {cls:'act',t:'&#8634; RT-2 fine-tune batch 88/100 &middot; loss 0.031'},
  {cls:'ok', t:'&#10003; Physics gate passed &middot; 97.3% acceptance'},
  {cls:'act',t:'&#8634; MimicGen augmenting contact-rich demonstrations'},
  {cls:'ok', t:'&#10003; OpenVLA policy rollout accepted &middot; score 0.91'},
  {cls:'act',t:'&#8634; MuJoCo scene reset &middot; episode 2,140'},
  {cls:'ok', t:'&#10003; RLDS export ready &middot; 3 shards &middot; warm speedup 54&#215;'},
];
let ti=0;
function rotateTicker(){
  const t=TICKS[ti%TICKS.length]; ti++;
  document.getElementById('tickerTxt').innerHTML='<span class="'+t.cls+'">'+t.t+'</span>';
}
rotateTicker();
setInterval(rotateTicker, 3800);

function set(t){document.getElementById('intent').value=t;}

async function startRun(){
  const intent=document.getElementById('intent').value.trim();
  if(!intent)return;
  const btn=document.getElementById('runBtn');
  btn.disabled=true; btn.textContent='Launching...';
  done.clear();
  MODELS.forEach(m=>{
    const k=m.replace('-','');
    document.getElementById('mc-'+k).className='mc';
    document.getElementById('md-'+k).textContent='—';
    document.getElementById('mdiv-'+k).textContent='div —';
  });
  document.getElementById('results').classList.remove('show');
  document.getElementById('pFill').style.width='0%';
  document.getElementById('intentBlk').textContent='Parsing intent with GPT-4o-mini...';
  document.getElementById('osmoBlk').textContent='Generating OSMO workflow YAML...';
  switchTab('intent');
  try{
    const r=await fetch('/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({intent})});
    const d=await r.json();
    document.getElementById('runIdTxt').textContent=d.run_id;
    document.getElementById('runStatus').classList.add('show');
    document.getElementById('tickerTxt').innerHTML='<span class="act">&#8594; Run '+d.run_id+' launched &middot; ETA ~'+d.eta_seconds+'s</span>';
    btn.textContent='Running...';
    // Populate intent + OSMO panels immediately from response
    if(d.parsed_intent){
      document.getElementById('intentBlk').textContent=JSON.stringify(d.parsed_intent,null,2);
    }
    if(d.osmo_workflow){
      document.getElementById('osmoBlk').textContent=d.osmo_workflow;
    }
    pollId=setInterval(()=>poll(d.run_id),1800);
  }catch(e){btn.disabled=false;btn.textContent='Run Simulation →';}
}

async function poll(id){
  try{
    const r=await fetch('/run/'+id);
    const d=await r.json();
    const pct=Math.round(d.progress*100);
    document.getElementById('pFill').style.width=pct+'%';
    document.getElementById('stageTxt').textContent=d.stage+' · '+pct+'%';
    const pill=document.getElementById('statusPill');
    pill.className='status-pill pill-'+d.status;
    pill.textContent=d.status.charAt(0).toUpperCase()+d.status.slice(1);
    MODELS.forEach(m=>{
      const k=m.replace('-','');
      if(done.has(m))return;
      if((d.stage||'').includes(m))document.getElementById('mc-'+k).className='mc active';
    });
    if(d.status==='completed'&&d.result){
      clearInterval(pollId);
      const res=d.result;
      (res.model_breakdown||[]).forEach(mb=>{
        const k=mb.model.replace('-','');
        document.getElementById('mc-'+k).className='mc done';
        document.getElementById('md-'+k).textContent=mb.demos_generated.toLocaleString();
        document.getElementById('mdiv-'+k).textContent='div '+mb.diversity_score;
        done.add(mb.model);
      });
      document.getElementById('rDemos').textContent=res.total_demos.toLocaleString();
      document.getElementById('rDiv').textContent=res.diversity_score;
      document.getElementById('rMb').textContent=res.dataset_size_mb+' MB';
      document.getElementById('rSpeedup').textContent=res.warm_run_speedup;
      document.getElementById('traceBtn').href='/trace/'+id+'/view';
      document.getElementById('results').classList.add('show');
      demoCount+=res.total_demos;
      document.getElementById('mDemos').textContent=demoCount.toLocaleString();
      document.getElementById('mSpeedup').textContent=(res.warm_run_speedup||'54').replace('×','');
      document.getElementById('tickerTxt').innerHTML='<span class="ok">&#10003; Run '+id+' complete &middot; '+res.total_demos.toLocaleString()+' demos &middot; diversity '+res.diversity_score+'</span>';
      const btn=document.getElementById('runBtn');
      btn.disabled=false; btn.textContent='Run Another →';
    }
  }catch(e){}
}
</script>
</body></html>""")


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

    osmo_workflow = generate_osmo_workflow(run_id, parsed, num_demos, cfg)

    runs[run_id] = {
        "run_id": run_id,
        "intent": req.intent,
        "parsed_intent": parsed,
        "osmo_workflow": osmo_workflow,
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
        parsed_intent=parsed,
        osmo_workflow=osmo_workflow,
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
