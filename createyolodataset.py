import glob
import os


def breakline(line):
    img, _, classe, x, y, w, h = line.split('\t')

    return int(img), int(classe), float(x), float(y), float(w), float(h)


def clip(n):
    if n > 1:
        n = 1
    if n < 0:
        n = 0

    return n


if __name__ == '__main__':
    list_seqs = glob.glob('camel/images/*/')
    train_images = open('camel/train.txt', 'w+')
    valid_images = open('camel/val.txt', 'w+')
    test_images = open('camel/test.txt', 'w+')

    class_freq = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 17: 0}
    for seq in list_seqs:
        seq_name = seq.split('/')[-2]
        # os.mkdir('camel/labels/{}'.format(seq_name))
        # os.mkdir('camel/labels/{}/IR'.format(seq_name))
        try:
            with open('{}{}-IR.txt'.format(seq, seq_name)) as annotations:
                lines = annotations.read().splitlines()
                current_img = 1
                current_output = open('camel/labels/{}/IR/{}.txt'.format(seq_name, str(current_img).zfill(6)), 'w+')

                try:
                    open('camel/images/{}/IR/{}.jpg'.format(seq_name, str(current_img).zfill(6)))
                    extension = 'jpg'
                except FileNotFoundError:
                    extension = 'png'

                lines.sort()

                for line in lines:
                    if line == '':
                        continue

                    img, classe, x, y, w, h = breakline(line)

                    class_freq[classe - 1] = class_freq[classe - 1] + 1
                    if classe > 3:
                        continue

                    if x < 0:
                        x = 0

                    if current_img != img:
                        if seq_name in ['Seq4', 'Seq5', 'Seq30']:
                            test_images.write('camel/images/{}/IR/{}.{}\n'.format(seq_name, str(current_img).zfill(6),
                                                                                  extension))
                        elif seq_name == 'Seq29':
                            valid_images.write('camel/images/{}/IR/{}.{}\n'.format(seq_name, str(current_img).zfill(6),
                                                                                   extension))
                        else:
                            train_images.write('camel/images/{}/IR/{}.{}\n'.format(seq_name, str(current_img).zfill(6),
                                                                                   extension))
                        current_output.close()
                        current_img = img
                        current_output = open('camel/labels/{}/IR/{}.txt'.format(seq_name,
                                                                                     str(current_img).zfill(6)), 'w+')

                    x_center = clip((x + w / 2) / 336)
                    y_center = clip((y + h / 2) / 256)

                    w = clip(w / 336)
                    h = clip(h / 256)

                    current_output.write('{} {} {} {} {}\n'.format(classe - 1, x_center, y_center, w, h))
            current_output.close()

        except FileNotFoundError:
            print("Sequenza senza annotazioni")

    print(class_freq)
    train_images.close()
    valid_images.close()
    test_images.close()
