start:
	mkdir -p tmp
	COMPOSE_PROJECT_NAME=gratheon docker compose -f docker-compose.dev.yml up --build
stop:
	COMPOSE_PROJECT_NAME=gratheon docker compose -f docker-compose.dev.yml down
run:
	npm run dev

# deploy-clean:
# 	ssh root@gratheon.com 'rm -rf /www/models-yolov5/;'

deploy-copy:
	rsync -av -e ssh * root@gratheon.com:/www/models-yolov5/

deploy-run:
	ssh root@gratheon.com 'chmod +x /www/models-yolov5/restart.sh'
	ssh root@gratheon.com 'bash /www/models-yolov5/restart.sh'

deploy:
	# make deploy-clean
	make deploy-copy
	make deploy-run

.PHONY: deploy
