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
    -e SERVER_ADDRESS='https://app.supervise.ly' \
    -e API_TOKEN='HKPFXmLJaQ4e1ZLzDbCv2WHEsbjjLOAZE7kKb14JKZFZVpQDC1YH0rYfJSPpL34y0wO4LuGrSXn6l7z1RU5TeZw6UIpkONOCP8LmijEcBYrfJ12NP2T76RbS75Xvx2fm' \
    -v ${PWD}/..:/alex_work \
    -v ${PWD}/../../alex_data:/alex_data \
    -v /home/andrew/alex_work/supervisely_py/supervisely_lib:/alex_work/supervisely_lib \
    -v /opt/pycharm:/pycharm \
    -v /home/andrew/pycharm-settings/smtool_gui:/root/.PyCharmCE2018.2 \
    -v /home/andrew/pycharm-settings/smtool_gui__idea:/workdir/.idea \
    --add-host="app.supervise.ly:136.243.97.171" \
    yolo5 bash
