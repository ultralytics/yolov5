# Dataset utils and dataloaders

from utils.general_polygon import xyxyxyxyn2xyxyxyxy, LOGGER, colorstr
from utils.datasets import *
help_url = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes
num_threads = min(8, os.cpu_count())  # number of multiprocessing threads
# Ancillary functions with polygon anchor boxes-------------------------------------------------------------------------------------------
def polygon_create_dataloader(path, imgsz, batch_size, stride, single_cls=False, hyp=None, augment=False, cache=False, pad=0.0,
                      rect=False, rank=-1, workers=8, image_weights=False, quad=False, prefix='', shuffle=False):
    if rect and shuffle:
        LOGGER.warning('WARNING: --rect is incompatible with DataLoader shuffle, setting shuffle=False')
        shuffle = False
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        dataset = Polygon_LoadImagesAndLabels(path, imgsz, batch_size,
                                      augment=augment,  # augmentation
                                      hyp=hyp,  # hyperparameters
                                      rect=rect,  # rectangular batches
                                      cache_images=cache,
                                      single_cls=single_cls,
                                      stride=int(stride),
                                      pad=pad,
                                      image_weights=image_weights,
                                      prefix=prefix)

    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    loader = DataLoader if image_weights else InfiniteDataLoader  # only DataLoader allows for attribute updates
    return loader(dataset,
                  batch_size=batch_size,
                  shuffle=shuffle and sampler is None,
                  num_workers=nw,
                  sampler=sampler,
                  pin_memory=True,
                  collate_fn=Polygon_LoadImagesAndLabels.collate_fn4 if quad else Polygon_LoadImagesAndLabels.collate_fn), dataset



class Polygon_LoadImagesAndLabels(Dataset):  # for training/testing
    """
        Polygon_LoadImagesAndLabels for polygon boxes
    """
    def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
                 cache_images=False, single_cls=False, stride=32, pad=0.0, prefix=''):
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.path = path
        
        # albumentation
        self.albumentations = Albumentations() if augment else None

        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                    # f = list(p.rglob('**/*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p, 'r') as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                        # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise Exception(f'{prefix}{p} does not exist')
            self.img_files = sorted([x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in img_formats])
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in img_formats])  # pathlib
            assert self.img_files, f'{prefix}No images found'
        except Exception as e:
            raise Exception(f'{prefix}Error loading data from {path}: {e}\nSee {help_url}')

        # Check cache
        self.label_files = img2label_paths(self.img_files)  # labels
        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix('.cache')  # cached labels
        if cache_path.is_file():
            cache, exists = torch.load(cache_path), True  # load
            if cache.get('hash') != get_hash(self.label_files + self.img_files):
                cache, exists = self.cache_labels(cache_path, prefix), False  # re-cache
        else:
            cache, exists = self.cache_labels(cache_path, prefix), False  # cache

        # Display cache
        nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupted, total
        if exists:
            d = f"Scanning '{cache_path}' images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupted"
            tqdm(None, desc=prefix + d, total=n, initial=n)  # display cache results
        assert nf > 0 or not augment, f'{prefix}No labels in {cache_path}. Can not train without labels. See {help_url}'

        # Read cache
        cache.pop('hash')  # remove hash
        cache.pop('version')  # remove version
        labels, shapes, self.segments = zip(*cache.values())
        self.labels = list(labels)
        self.shapes = np.array(shapes, dtype=np.float64)
        self.img_files = list(cache.keys())  # update
        self.label_files = img2label_paths(cache.keys())  # update
        if single_cls:
            for x in self.labels:
                x[:, 0] = 0

        n = len(shapes)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = range(n)

        # Rectangular Training
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # image wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int) * stride

        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        self.imgs, self.img_npy = [None] * n, [None] * n
        if cache_images:
            gb = 0  # Gigabytes of cached images
            self.img_hw0, self.img_hw = [None] * n, [None] * n
            results = ThreadPool(num_threads).imap(lambda x: load_image(*x), zip(repeat(self), range(n)))
            pbar = tqdm(enumerate(results), total=n)
            for i, x in pbar:
                self.imgs[i], self.img_hw0[i], self.img_hw[i] = x  # img, hw_original, hw_resized = load_image(self, i)
                gb += self.imgs[i].nbytes
                pbar.desc = f'{prefix}Caching images ({gb / 1E9:.1f}GB)'
            pbar.close()

    def cache_labels(self, path=Path('./labels.cache'), prefix=''):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        nm, nf, ne, nc = 0, 0, 0, 0  # number missing, found, empty, corrupt
        desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels..."
        with Pool(num_threads) as pool:
            pbar = tqdm(pool.imap_unordered(polygon_verify_image_label, zip(self.img_files, self.label_files, repeat(prefix))),
                        desc=desc, total=len(self.img_files))
            for im_file, l, shape, segments, nm_f, nf_f, ne_f, nc_f in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x[im_file] = [l, shape, segments]
                pbar.desc = f"{desc}{nf} found, {nm} missing, {ne} empty, {nc} corrupted"

        pbar.close()
        if nf == 0:
            LOGGER.info(f'{prefix}WARNING: No labels found in {path}. See {help_url}')
        x['hash'] = get_hash(self.label_files + self.img_files)
        x['results'] = nf, nm, ne, nc, len(self.img_files)
        x['version'] = 0.2  # cache version
        try:
            torch.save(x, path)  # save cache for next time
            LOGGER.info(f'{prefix}New cache created: {path}')
        except Exception as e:
            LOGGER.info(f'{prefix}WARNING: Cache directory {path.parent} is not writeable: {e}')  # path not writeable
        return x

    def __len__(self):
        return len(self.img_files)

    # def __iter__(self):
    #     self.count = -1
    #     print('ran dataset iter')
    #     #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
    #     return self
    
    def __getitem__(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']
        if mosaic:
            # Load mosaic
            img, labels = polygon_load_mosaic(self, index)
            shapes = None

            # MixUp https://arxiv.org/pdf/1710.09412.pdf
            if random.random() < hyp['mixup']:
                img2, labels2 = polygon_load_mosaic(self, random.randint(0, self.n - 1))
                r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
                img = (img * r + img2 * (1 - r)).astype(np.uint8)
                labels = np.concatenate((labels, labels2), 0)

        else:
            # Load image
            img, (h0, w0), (h, w) = load_image(self, index)

            # Letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            labels = self.labels[index].copy() 
            if labels.size:  # normalized format to pixel xyxyxyxy format
                labels[:, 1:] = xyxyxyxyn2xyxyxyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

        if self.augment:
            # Augment imagespace
            if not mosaic:
                img, labels = polygon_random_perspective(img, labels,
                                                         degrees=hyp['degrees'],
                                                         translate=hyp['translate'],
                                                         scale=hyp['scale'],
                                                         shear=hyp['shear'],
                                                         perspective=hyp['perspective'])

            # Augment colorspace
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])
            # Polygon does not support cutouts

        nL = len(labels)  # number of labels
        if nL:
            labels[:, 2::2] /= img.shape[0]  # normalized height 0-1
            labels[:, 1::2] /= img.shape[1]  # normalized width 0-1

        if self.augment:
            # albumentation
            img = self.albumentations(img)
            
            # flip up-down for all y
            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if nL:
                    labels[:, 2::2] = 1 - labels[:, 2::2]

            # flip left-right for all x
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if nL:
                    labels[:, 1::2] = 1 - labels[:, 1::2]
        # original label shape is (nL, 9), add one column for target image index for build_targets()
        labels_out = torch.zeros((nL, 10))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out, self.img_files[index], shapes

    # Polygon does not support collate_fn4
    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes
    
    
def polygon_load_mosaic(self, index):
    # loads images in a 4-mosaic, with polygon boxes

    labels4, segments4 = [], []
    s = self.img_size
    yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y
    indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(self, index)

        # place img in img4
        if i == 0:  # top left
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        labels, segments = self.labels[index].copy(), self.segments[index].copy()
        if labels.size:
            # from normalized, unpadded xyxyxyxy to pixel, padded xyxyxyxy format
            labels[:, 1:] = xyxyxyxyn2xyxyxyxy(labels[:, 1:], w, h, padw, padh)
            segments = [xyxyxyxyn2xyxyxyxy(x, w, h, padw, padh) for x in segments]
        labels4.append(labels)
        segments4.extend(segments)

    # Concat/clip labels
    labels4 = np.concatenate(labels4, 0)
    for x in (labels4[:, 1:], *segments4):
        np.clip(x, 0, 2 * s, out=x)  # inplace clip when using polygon_random_perspective()

    # Augment
    img4, labels4 = polygon_random_perspective(img4, labels4, segments4,
                                               degrees=self.hyp['degrees'],
                                               translate=self.hyp['translate'],
                                               scale=self.hyp['scale'],
                                               shear=self.hyp['shear'],
                                               perspective=self.hyp['perspective'],
                                               border=self.mosaic_border,
                                               mosaic=True)  # border to remove

    return img4, labels4


def polygon_load_mosaic9(self, index):
    # loads images in a 9-mosaic, with polygon boxes

    labels9, segments9 = [], []
    s = self.img_size
    indices = [index] + random.choices(self.indices, k=8)  # 8 additional image indices
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(self, index)

        # place img in img9
        if i == 0:  # center
            img9 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            h0, w0 = h, w
            c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
        elif i == 1:  # top
            c = s, s - h, s + w, s
        elif i == 2:  # top right
            c = s + wp, s - h, s + wp + w, s
        elif i == 3:  # right
            c = s + w0, s, s + w0 + w, s + h
        elif i == 4:  # bottom right
            c = s + w0, s + hp, s + w0 + w, s + hp + h
        elif i == 5:  # bottom
            c = s + w0 - w, s + h0, s + w0, s + h0 + h
        elif i == 6:  # bottom left
            c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
        elif i == 7:  # left
            c = s - w, s + h0 - h, s, s + h0
        elif i == 8:  # top left
            c = s - w, s + h0 - hp - h, s, s + h0 - hp

        padx, pady = c[:2]
        x1, y1, x2, y2 = [max(x, 0) for x in c]  # allocate coords

        # Labels
        labels, segments = self.labels[index].copy(), self.segments[index].copy()
        if labels.size:
            # from normalized, unpadded xyxyxyxy to pixel, padded xyxyxyxy format
            labels[:, 1:] = xyxyxyxyn2xyxyxyxy(labels[:, 1:], w, h, padx, pady)  
            segments = [xyxyxyxyn2xyxyxyxy(x, w, h, padx, pady) for x in segments]
        labels9.append(labels)
        segments9.extend(segments)

        # Image
        img9[y1:y2, x1:x2] = img[y1 - pady:, x1 - padx:]  # img9[ymin:ymax, xmin:xmax]
        hp, wp = h, w  # height, width previous

    # Offset
    yc, xc = [int(random.uniform(0, s)) for _ in self.mosaic_border]  # mosaic center x, y
    img9 = img9[yc:yc + 2 * s, xc:xc + 2 * s]

    # Concat/clip labels
    labels9 = np.concatenate(labels9, 0)
    labels9[:, 1::2] -= xc
    labels9[:, 2::2] -= yc
    c = np.array([xc, yc])  # centers
    segments9 = [x - c for x in segments9]

    for x in (labels9[:, 1:], *segments9):
        np.clip(x, 0, 2 * s, out=x)  # inplace clip when using polygon_random_perspective()

    # Augment
    img9, labels9 = polygon_random_perspective(img9, labels9, segments9,
                                               degrees=self.hyp['degrees'],
                                               translate=self.hyp['translate'],
                                               scale=self.hyp['scale'],
                                               shear=self.hyp['shear'],
                                               perspective=self.hyp['perspective'],
                                               border=self.mosaic_border,
                                               mosaic=True)  # border to remove

    return img9, labels9


def polygon_verify_image_label(params):
    # Verify one image-label pair
    im_file, lb_file, prefix = params
    nm, nf, ne, nc = 0, 0, 0, 0  # number missing, found, empty, corrupt
    try:
        # verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        segments = []  # instance segments
        assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
        assert im.format.lower() in img_formats, f'invalid image format {im.format}'

        # verify labels
        if os.path.isfile(lb_file):
            nf = 1  # label found
            l = np.loadtxt(lb_file, dtype=np.float32)
            with open(lb_file) as f:
                l = np.array([x.split() for x in f.readlines() if len(x)>1], dtype=np.float32)  # labels # ata_landmark_change
            # l=np.zeros((ll.shape[0], 9))
            # l[:,1:] =ll[:,5:]
            if len(l.shape)==1: l = l[None, :]
            segments = [l[0][1:].reshape(-1, 2) for x in l]  # ((x1, y1), (x2, y2), ...)
            if len(l):
                assert l.shape[1] == 9, 'labels require 9 columns each'
                # Common out following lines to enable: polygon corners can be out of images
                # assert (l >= 0).all(), 'negative labels'
                # assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels'
                assert np.unique(l, axis=0).shape[0] == l.shape[0], 'duplicate labels'
            else:
                ne = 1  # label empty
                l = np.zeros((0, 9), dtype=np.float32)
        else:
            nm = 1  # label missing
            l = np.zeros((0, 9), dtype=np.float32)
        return im_file, l, shape, segments, nm, nf, ne, nc
    except Exception as e:
        nc = 1
        LOGGER.info(f'{prefix}WARNING: Ignoring corrupted image and/or label {im_file}: {e}')
        return [None] * 4 + [nm, nf, ne, nc]


class Albumentations:
    # Polygon YOLOv5 Albumentations class (optional, only used if package is installed)
    def __init__(self):
        self.transform = None
        try:
            import albumentations as A

            self.transform = A.Compose([
                A.MedianBlur(p=0.05),
                A.ToGray(p=0.1),
                A.RandomBrightnessContrast(p=0.35),
                A.CLAHE(p=0.2),
                A.InvertImg(p=0.3)],)
                # Not support for any position change to image

            LOGGER.info(colorstr('albumentations: ') + ', '.join(f'{x}' for x in self.transform.transforms if x.p))
        except ImportError:  # package not installed, skip
            pass
        except Exception as e:
            LOGGER.info(colorstr('albumentations: ') + f'{e}')

    def __call__(self, im, p=1.0):
        if self.transform and random.random() < p:
            new = self.transform(image=im)  # transformed
            im = new['image']
        return im


def polygon_random_perspective(img, targets=(), segments=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0,
                       border=(0, 0), mosaic=False):
    # """
    #     torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    #     targets = [cls, xyxyxyxy]
    # """
    #     To restrict the polygon boxes within images
    #     def restrict(img, new, shape0, padding=(0, 0, 0, 0)):
    #         height, width = shape0
    #         top0, bottom0, left0, right0 = np.ceil(padding[0]), np.floor(padding[1]), np.floor(padding[2]), np.ceil(padding[3])
    #         # keep the original shape of image
    #         if (height/width) < ((height+bottom0+top0)/(width+left0+right0)):
    #             dw = int((height+bottom0+top0)/height*width)-(width+left0+right0)
    #             top, bottom, left, right = map(int, (top0, bottom0, left0+dw/2, right0+dw/2))
    #         else:
    #             dh = int((width+left0+right0)*height/width)-(height+bottom0+top0)
    #             top, bottom, left, right = map(int, (top0+dh/2, bottom0+dh/2, left0, right0))
    #         img = cv2.copyMakeBorder(img, bottom, top, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    #         img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    #         w_r, h_r = width/(width+left+right), height/(height+bottom+top)
    #         new[:, 0::2] = (new[:, 0::2]+left)*w_r
    #         new[:, 1::2] = (new[:, 1::2]+bottom)*h_r
    #         return img, new
            
    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    image_transformed = False

    # Transform label coordinates
    n = len(targets)
    if n:
        # if using segments: please use general.py::polygon_segment2box
        # segment is unnormalized np.array([[(x1, y1), (x2, y2), ...], ...])
        # targets is unnormalized np.array([[class id, x1, y1, x2, y2, ...], ...])
        new = np.zeros((n, 8))
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, 1:].reshape(n * 4, 2)
        xy = xy @ M.T  # transform
        new = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine
        
        if not mosaic:
            # Compute Top, Bottom, Left, Right Padding to Include Polygon Boxes inside Image
            top = max(new[:, 1::2].max().item()-height, 0)
            bottom = abs(min(new[:, 1::2].min().item(), 0))
            left = abs(min(new[:, 0::2].min().item(), 0))
            right = max(new[:, 0::2].max().item()-width, 0)
            
            R2 = np.eye(3)
            r = min(height/(height+top+bottom), width/(width+left+right))
            R2[:2] = cv2.getRotationMatrix2D(angle=0., center=(0, 0), scale=r)
            M2 = T @ S @ R @ R2 @ P @ C  # order of operations (right to left) is IMPORTANT
            
            if (border[0] != 0) or (border[1] != 0) or (M2 != np.eye(3)).any():  # image changed
                if perspective:
                    img = cv2.warpPerspective(img, M2, dsize=(width, height), borderValue=(114, 114, 114))
                else:  # affine
                    img = cv2.warpAffine(img, M2[:2], dsize=(width, height), borderValue=(114, 114, 114))
                image_transformed = True
                new = np.zeros((n, 8))
                xy = np.ones((n * 4, 3))
                xy[:, :2] = targets[:, 1:].reshape(n * 4, 2)
                xy = xy @ M2.T  # transform
                new = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine
            # img, new = restrict(img, new, (height, width), (top, bottom, left, right))
        
        # Use the following two lines can result in slightly tilting for few labels.
        # new[:, 0::2] = new[:, 0::2].clip(0., width)
        # new[:, 1::2] = new[:, 1::2].clip(0., height)
        # If use following codes instead, can mitigate tilting problems, but result in few label exceeding problems.
        cx, cy = new[:, 0::2].mean(-1), new[:, 1::2].mean(-1)
        new[(cx>width)|(cx<-0.)|(cy>height)|(cy<-0.)] = 0.
        
        # filter candidates
        # 0.1 for axis-aligned rectangle, 0.01 for segmentation, so choose intermediate 0.08
        i = polygon_box_candidates(box1=targets[:, 1:].T * s, box2=new.T, area_thr=0.08) 
        targets = targets[i]
        targets[:, 1:] = new[i]
        
    if not image_transformed:
        M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
        if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
            if perspective:
                img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
            else:  # affine
                img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))
            image_transformed = True
        
    return img, targets


def polygon_box_candidates(box1, box2, wh_thr=3, ar_thr=20, area_thr=0.1, eps=1e-16):
    """
        box1(8,n), box2(8,n)
        Use the minimum bounding box as the approximation to polygon
        Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    """
    w1, h1 = box1[0::2].max(axis=0)-box1[0::2].min(axis=0), box1[1::2].max(axis=0)-box1[1::2].min(axis=0)
    w2, h2 = box2[0::2].max(axis=0)-box2[0::2].min(axis=0), box2[1::2].max(axis=0)-box2[1::2].min(axis=0)
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates










