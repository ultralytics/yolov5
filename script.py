import json
import os
import random

a = 0

fin1 = open("./custom_train/trainfs.txt", "a+")
fin2 = open("./custom_train/validfs.txt", "a+")

my_randoms = random.sample(range(0, 13000), 10400)
for file in os.listdir("../yolov3/custom_data/FaceSwap/images/"):
	a += 1
	path = os.path.join("../yolov3/custom_data/FaceSwap/images/",file)
	if a in my_randoms:  
		fin1.write(path + "\n")
	else:
		fin2.write(path + "\n")

fin1.close()
fin2.close()