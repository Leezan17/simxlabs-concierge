# SimXLabs — Simulation Concierge Knowledge Base

## What is SimXLabs?

SimXLabs is an AI orchestration platform for robotics simulation. It takes a plain-English description of a task and automatically generates large-scale, physics-verified training datasets for robot learning — without requiring any simulation expertise from the user.

SimXLabs is built by Leeza Nadeem and Ignacio Erazo, part of the Techstars Boulder 2026 cohort.

---

## How It Works

1. **Describe** — The user tells SimXLabs what they need in plain English (e.g. "bin picking, 10,000 demos, maximize diversity")
2. **Plan** — SimXLabs parses the intent, selects the right simulation engine, and builds an execution DAG (a step-by-step plan with dependencies)
3. **Orchestrate** — The system routes the task across four foundation models simultaneously: Pi0, RT-2, OpenVLA, and MimicGen
4. **Verify** — Every output is physics-checked. Bad demos are removed automatically
5. **Deliver** — A validated, diversity-rich dataset is returned, ready to drop into training

---

## Foundation Models SimXLabs Uses

- **Pi0** (Physical Intelligence) — General-purpose physical robot policies. Strong on manipulation tasks
- **RT-2** (Google DeepMind) — Vision-language-action model. Excels at instruction-following with visual context
- **OpenVLA** (Open-source) — Open Visual-Language-Action model. Good coverage of diverse environments
- **MimicGen** (NVIDIA) — Data synthesis via human demonstration augmentation. Great for precise, repeatable tasks

Using all four simultaneously reduces individual model bias and produces richer, more generalizable training data.

---

## Supported Task Types

| Task | Typical Demo Count | Best Engine | Notes |
|------|-------------------|-------------|-------|
| Bin Picking | 10,000 | Isaac Sim | High trajectory diversity |
| Peg Insertion | 8,000 | MuJoCo | Sub-millimeter precision |
| Door Opening | 12,000 | Isaac Sim | Varied handle types |
| Cloth Folding | 6,000 | Genesis | Deformable object dynamics |
| Custom Task | 10,000 | Varies | Describe in plain English |

---

## Key Features

**Self-healing** — If a simulation node fails mid-run, SimXLabs automatically retries with adjusted parameters. Users never see partial outputs.

**Physics verification** — Every demo passes constraint checks before inclusion. Physically impossible motions are filtered out automatically.

**Semantic caching** — Once a workflow has been run, similar future requests hit the cache. Warm runs can be 40–80× faster.

**Full traceability** — Every run produces a detailed trace showing routing decisions, model contributions, verification outcomes, and cache hits. Each trace has a unique URL.

**Reusable workflows** — Workflows compound in value. Each run makes the next one faster and richer.

---

## What Users Get Back

After a run completes, SimXLabs returns:
- Total verified demonstrations generated
- Diversity score (0–1, higher = more varied robot strategies)
- Dataset size (MB) in HDF5 + RLDS format
- Breakdown by foundation model
- A trace link with full DAG execution log
- A download URL for the dataset

---

## How to Start a Run (as the Concierge)

When a user asks to generate simulation data, trigger the `start_simulation` action with their plain-English description. You don't need to translate it — SimXLabs handles intent parsing internally.

Example triggers:
- "Run bin picking with 10,000 demos"
- "Generate door opening scenarios across different handle types"
- "I need peg insertion data with sub-millimeter precision"
- "Create training data for cloth folding"

After triggering, give the user their run ID and let them know you'll check back on the status.

---

## Status Checks

Use `check_run_status` with the run ID to get current progress. The run goes through stages:
1. Queued
2. Parsing intent
3. Compiling simulation environment
4. Planning execution DAG
5. Sampling — [Model Name] (×4)
6. Verifying and deduplicating
7. Done

---

## Common Questions

**Q: How long does a run take?**
A: Typically 1–3 minutes for the pilot. Production runs targeting millions of demos run on distributed cloud infrastructure.

**Q: What format is the output?**
A: HDF5 and RLDS — both are standard formats compatible with major robot learning frameworks (LeRobot, Open X-Embodiment, etc.).

**Q: Can I run the same task twice?**
A: Yes — and the second run will be dramatically faster thanks to semantic caching. Typical speedup is 40–80×.

**Q: What if my task isn't one of the four standard types?**
A: Just describe it in plain English. SimXLabs will parse your intent and select the best configuration automatically.

**Q: Who built SimXLabs?**
A: Leeza Nadeem (CS, University of Rochester — formerly Apple, NVIDIA Omniverse, Stanford AI Lab) and Ignacio Erazo (Ph.D. Operations Research, Georgia Tech — formerly Apple, Amazon Robotics, $70M+ annual impact).

---

## Contact & Links

- Website: simxlabs.com
- Demo: simxlabs.com/demo/foundation-models
- Email: contact@simxlabs.com
- Techstars Boulder 2026
