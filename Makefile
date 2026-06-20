.PHONY: install run tunnel test

install:
	cd api && pip install -r requirements.txt

run:
	cd api && uvicorn main:app --reload --port 8000

# Expose local server publicly for Convai to reach
tunnel:
	ngrok http 8000

# Quick smoke test — run these after `make run` in another terminal
test:
	@echo "\n── Health check ──"
	curl -s http://localhost:8000/health | python3 -m json.tool

	@echo "\n── Start a run ──"
	curl -s -X POST http://localhost:8000/run \
	  -H "Content-Type: application/json" \
	  -d '{"intent": "bin picking, 10000 demos, maximize trajectory diversity"}' \
	  | python3 -m json.tool
