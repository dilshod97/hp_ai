.PHONY: up down logs ps pull-models ingest-laws ingest-reports health rebuild

# docker-compose (v1) yoki docker compose (v2) — qaysi bori ishlaydi
COMPOSE := $(shell command -v docker-compose 2> /dev/null || echo "docker compose")

up:
	$(COMPOSE) up -d --build

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f --tail=200 backend

ps:
	$(COMPOSE) ps

rebuild:
	$(COMPOSE) build --no-cache backend && $(COMPOSE) up -d backend

# Ollama (server hostda ishlaydi)
pull-models:
	ollama pull gpt-oss:20b
	ollama pull bge-m3

ingest-laws:
	$(COMPOSE) exec backend python scripts/ingest.py --target laws --path /data/laws

ingest-reports:
	$(COMPOSE) exec backend python scripts/ingest.py --target reports --path /data/reports

health:
	curl -s http://localhost:8000/api/health | python -m json.tool
