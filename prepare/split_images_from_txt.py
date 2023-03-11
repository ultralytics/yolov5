import os

import tqdm

modes = ['train', 'val', 'test']
data = []
for mode in modes:
    mode_path = f'../dataset/images/{mode}'
    if not os.path.exists(mode_path):
        os.mkdir(mode_path)
        print(f'create {mode_path}')
    path = f'../dataset/{mode}.txt'
    for line in open(path):
        data.append(line.replace('\n', ''))
    for a in tqdm.tqdm(data):
        try:
            os.system(f'cp ../dataset/preimages/{a}.jpg ../dataset/images/{mode}/{a}.jpg')
        except OSError:
            print(f'{a} is not exist')
    data.clear()
