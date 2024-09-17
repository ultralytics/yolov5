start:
	COMPOSE_PROJECT_NAME=gratheon docker compose -f docker-compose.dev.yml up
stop:
	COMPOSE_PROJECT_NAME=gratheon docker compose -f docker-compose.dev.yml down
