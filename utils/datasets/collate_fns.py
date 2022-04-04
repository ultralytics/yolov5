import torch.nn.functional as F
import torch
import random
# #####################################

def collate_fn(batch):
    im, label, path, shapes = zip(*batch)  # transposed
    for i, lb in enumerate(label):
        lb[:, 0] = i  # add target image index for build_targets()
    return torch.stack(im, 0), torch.cat(label, 0), path, shapes
# #####################################

def collate_fn4(batch):
    img, label, path, shapes = zip(*batch)  # transposed
    n = len(shapes) // 4
    im4, label4, path4, shapes4 = [], [], path[:n], shapes[:n]

    ho = torch.tensor([[0.0, 0, 0, 1, 0, 0]])
    wo = torch.tensor([[0.0, 0, 1, 0, 0, 0]])
    s = torch.tensor([[1, 1, 0.5, 0.5, 0.5, 0.5]])  # scale
    for i in range(n):  # zidane torch.zeros(16,3,720,1280)  # BCHW
        i *= 4
        if random.random() < 0.5:
            im = F.interpolate(img[i].unsqueeze(0).float(), scale_factor=2.0, mode='bilinear', align_corners=False)[
                0].type(img[i].type())
            lb = label[i]
        else:
            im = torch.cat((torch.cat((img[i], img[i + 1]), 1), torch.cat((img[i + 2], img[i + 3]), 1)), 2)
            lb = torch.cat((label[i], label[i + 1] + ho, label[i + 2] + wo, label[i + 3] + ho + wo), 0) * s
        im4.append(im)
        label4.append(lb)

    for i, lb in enumerate(label4):
        lb[:, 0] = i  # add target image index for build_targets()

    return torch.stack(im4, 0), torch.cat(label4, 0), path4, shapes4
# #####################################