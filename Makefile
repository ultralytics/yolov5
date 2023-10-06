start:
	COMPOSE_PROJECT_NAME=gratheon docker compose -f docker-compose.dev.yml --verbose up --build
stop:
	COMPOSE_PROJECT_NAME=gratheon docker compose -f docker-compose.dev.yml down

deploy-copy:
	rsync -av -e ssh * root@gratheon.com:/www/models-bee-detector/

deploy-run:
	ssh root@gratheon.com 'chmod +x /www/models-bee-detector/restart.sh'
	ssh root@gratheon.com 'bash /www/models-bee-detector/restart.sh'

deploy:
	make deploy-copy
	make deploy-run
