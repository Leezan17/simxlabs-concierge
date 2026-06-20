# SimXLabs × Convai — Simulation Concierge Pilot

> Voice-first interface that lets anyone trigger a SimXLabs simulation run and get back verified robot training data — powered by Convai's avatar + External API.

---

## What this is

A FastAPI server that acts as the SimXLabs backend for the Convai integration:

- **Hybrid simulation**: real LLM-based intent parsing (Claude Haiku) + realistic simulated DAG execution
- **3 External API methods**: `start_simulation`, `check_run_status`, `get_run_summary`
- **Dark-themed trace viewer** at `/trace/{run_id}/view`
- **Auto-refreshing** while run is in progress

---

## Quick start

```bash
# 1. Install
cd api && pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env

# 3. Run
make run
# → API live at http://localhost:8000

# 4. Expose publicly (for Convai to reach)
make tunnel
# → Copy the ngrok https URL

# 5. Smoke test
make test
```

Then follow `convai/CONVAI-SETUP.md` to connect the character.

---

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Server status |
| POST | `/run` | Start a simulation run |
| GET | `/run/{run_id}` | Get run status + progress |
| GET | `/run/{run_id}/summary` | Get narration-ready result summary |
| GET | `/trace/{run_id}` | Get full DAG trace (JSON) |
| GET | `/trace/{run_id}/view` | Trace viewer (HTML) |

---

## Project structure

```
simxlabs-concierge/
├── api/
│   ├── main.py              # FastAPI app (all logic)
│   └── requirements.txt
├── convai/
│   ├── knowledge-bank.md    # Upload to Convai KB
│   ├── character-system-prompt.md  # Paste into character backstory
│   ├── external-api-methods.json   # 3 External API method configs
│   └── CONVAI-SETUP.md      # Step-by-step Convai setup
├── .env.example
├── Makefile
└── README.md
```

---

## Demo flow

1. User: *"I need 10,000 bin picking demonstrations with high trajectory diversity"*
2. Simra (Convai): triggers `start_simulation` → *"Started run A3F7B2, ETA ~75s"*
3. User: *"How's it going?"*
4. Simra: triggers `check_run_status` → *"64% complete, sampling with MimicGen"*
5. Run completes → Simra triggers `get_run_summary`
6. Simra: *"Done — 9,847 verified demos, diversity score 0.88, 24.6 MB. Warm runs will be 52× faster. Trace: [link]"*

---

## Next steps (post-pilot)

- [ ] Swap in-memory store for Redis or Postgres
- [ ] Connect real solver backends (Isaac Sim API, MuJoCo server)
- [ ] Add authentication (API keys per customer)
- [ ] Deploy to Railway / Fly.io for persistent public URL
- [ ] Add Pixel Streaming embed for full 3D experience
