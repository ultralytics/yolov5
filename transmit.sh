# send from server
scp -P $IP_N4090_SERVER_PORT -r zhr@$IP_ANTIS_PUBLIC_IP:~/project/yolov5 .

# send to server
scp -P $IP_N3080_SERVER_PORT -r ../yolov5 zhr@$IP_ANTIS_PUBLIC_IP:~/project/yolov5/