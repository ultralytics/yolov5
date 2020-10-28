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
    -e API_TOKEN='gbZyTptb4qG0G2BKWmrGK4h5SC35DcTnxn4VWVA1VtJtOZRIprr3TPF8RbOAtef0NsBDPQw7b61dIueBe41cE08P98hyQ2A4CusN4d8TglfOoDk1Eq4W7oQjejablolN' \
    -v ${PWD}/..:/alex_work \
    -v ${PWD}/../../alex_data:/alex_data \
    -v /home/andrew/alex_work/supervisely_py/supervisely_lib:/alex_data/supervisely_lib \
    -v /opt/pycharm:/pycharm \
    -v /home/andrew/pycharm-settings/smtool_gui:/root/.PyCharmCE2018.2 \
    -v /home/andrew/pycharm-settings/smtool_gui__idea:/workdir/.idea \
    --add-host="app.supervise.ly:136.243.97.171" \
    docker.deepsystems.io/supervisely/five/base-py:6.0.17 bash

#pip install --upgrade docker to run script