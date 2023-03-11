import os

modes = ['train', 'val', 'test']
images_path = '../dataset/images'
pre_labels_path = '../dataset/prelabels'
post_labels_path = '../dataset/labels'

for mode in modes:
    mode_images_path = os.path.join(images_path, mode)
    mode_labels_path = os.path.join(post_labels_path, mode)
    if not os.path.exists(mode_labels_path):
        os.mkdir(os.path.join(mode_labels_path))
    for name in os.listdir(mode_images_path):
        name = name.replace('.jpg', '.txt')
        pre_label_path = os.path.join(pre_labels_path, name)
        try:
            with open(pre_label_path) as f:
                content = f.read()
                new_label_path = os.path.join(mode_labels_path, name)
            with open(new_label_path, 'w+') as g:
                g.write(content)
        except OSError:
            print(f'{pre_label_path} is not exist')
