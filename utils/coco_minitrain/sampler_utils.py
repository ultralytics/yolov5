'''
Ref: https://github.com/giddyyupp/coco-minitrain
'''

def get_coco_object_size_info(dataset_train,
                              areaRng = [32 ** 2, 96 ** 2, 1e5 ** 2]):
    size_dict = {}

    for k, v in dataset_train.coco.anns.items():
        # aa = Counter(v)
        area = v['bbox'][2] * v['bbox'][3]
        cat = v['category_id']
        if area < areaRng[0]:
            kk = str(cat) + "_S"
        elif area < areaRng[1]:
            kk = str(cat) + "_M"
        else:
            kk = str(cat) + "_L"

        # update counts
        if kk in size_dict:
            size_dict[kk] += 1
        else:
            size_dict[kk] = 1

    return size_dict


def get_coco_class_object_counts(dataset_train):
    annot_dict = {}

    for k, v in dataset_train.coco.catToImgs.items():
        # aa = Counter(v)
        annot_dict[k] = len(v)

    return annot_dict
