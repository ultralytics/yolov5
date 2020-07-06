import cv2


def breakline(line):
    classe, x_center, y_center, w, h = line.split(" ")

    return int(classe), float(x_center), float(y_center), float(w), float(h)


img = cv2.imread(r'camel\images\Seq15\IR\000053.jpg')

with open(r'camel\labels\Seq15\IR\000053.txt') as f:
    lines = f.read().splitlines()

    for line in lines:
        classe, x_center, y_center, w, h = breakline(line)

        w *= img.shape[1]
        h *= img.shape[0]

        u_x_center = x_center * img.shape[1]
        u_y_center = y_center * img.shape[0]

        tl_x = int(u_x_center - w / 2)
        tl_y = int(u_y_center - h / 2)
        br_x = int(tl_x + w)
        br_y = int(tl_y + h)

        img = cv2.rectangle(img, (tl_x, tl_y), (br_x, br_y), (255, 0, 0), 2)


cv2.imshow('out', img)
cv2.waitKey(0)
cv2.destroyAllWindows()