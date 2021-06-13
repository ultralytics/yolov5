FROM ultralytics/yolov5:latest

# Copy contents
COPY . /usr/src/app

RUN pip install -r requirements.txt
RUN pip install wandb
RUN wandb login e30bdfb905a3cf11769d005d2fc0b6d87cb67402

CMD bash -c "sh /usr/src/app/run.sh & sleep 5 && tail -F /dev/null"