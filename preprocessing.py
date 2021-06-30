import yaml
from glob import glob
from sklearn.model_selection import train_test_split

img_list = glob("./data/export/images/*.jpg")
img_list = [path.replace("data/", "") for path in img_list]
print(len(img_list))

train_img_list, val_img_list = train_test_split(
    img_list, test_size=0.2, random_state=2000
)
print(len(train_img_list), len(val_img_list))

with open("./data/train.txt", "w") as f:
    f.write("\n".join(train_img_list) + "\n")

with open("./data/val.txt", "w") as f:
    f.write("\n".join(val_img_list) + "\n")

with open("./data/data.yaml", "r") as f:
    data = yaml.safe_load(f)
    print(data)

data["train"] = "./data/train.txt"
data["val"] = "./data/val.txt"

with open("./data/data.yaml", "w") as f:
    yaml.dump(data, f)
    print(data)
