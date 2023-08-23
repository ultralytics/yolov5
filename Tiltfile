docker_compose('docker-compose.dev.yml',project_name="gratheon")
docker_build('local/models-yolov5', '.',
	live_update = [
    # Sync local files into the container.
    sync('.', '/app/'),

    # Re-run npm install whenever package.json changes.
    run('pip install -r requirements.txt', trigger='requirements.txt'),

    # Restart the process to pick up the changed files.
    restart_container()
  ])