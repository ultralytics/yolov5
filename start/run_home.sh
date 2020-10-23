docker run \
    --rm \
    -ti \
    --net=host \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v ~/.Xauthority:/root/.Xauthority \
    --entrypoint="" \
    --shm-size='1G' \
    -e PYTHONUNBUFFERED='1' \
    -v ${PWD}/..:/alex_work \
    -v ${PWD}/../../alex_data:/alex_data \
    -v /opt/pycharm:/pycharm \
    -v /home/andrew/pycharm-settings/smtool_gui:/root/.PyCharmCE2018.2 \
    -v /home/andrew/pycharm-settings/smtool_gui__idea:/workdir/.idea \
    yolo5 bash