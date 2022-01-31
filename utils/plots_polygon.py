from utils.plots import *

# Ancillary functions with polygon anchor boxes-------------------------------------------------------------------------------------------

def polygon_plot_one_box(x, im, color=(128, 128, 128), label=None, line_thickness=3):
    """
        Plots one bounding box on image 'im' using OpenCV
        im is np.array with shape (W, H, Ch), x is pixel-level xyxyxyxy
    """
    
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to polygon_plot_one_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    c = x.cpu().numpy().reshape(-1, 1, 2).astype(np.int32) if isinstance(x, torch.Tensor) else np.array(x).reshape(-1, 1, 2).astype(np.int32)
    cv2.polylines(im, pts=[c], isClosed=True, color=color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c1 = (int(c[:, 0, 0].mean()), int(c[:, 0, 1].mean()))
        c2 = (int(c1[0] + t_size[0]), int(c1[1] - t_size[1] - 3))
        im_origin = im.copy()
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        alpha = 0.5    # opacity of 0.5
        cv2.addWeighted(im, alpha, im_origin, 1 - alpha, 0, im)


def polygon_plot_one_box_PIL(x, im, color=(128, 128, 128), label=None, line_thickness=None):
    # Plots one bounding box on image 'im' using PIL
    im = Image.fromarray(im)
    draw = ImageDraw.Draw(im)
    line_thickness = line_thickness or max(int(min(im.size) / 200), 2)
    box = x.cpu().numpy().ravel().astype(np.int32) if isinstance(x, torch.Tensor) else np.array(x).ravel().astype(np.int32)
    draw.polygon(box.tolist(), outline=color)  # plot
    if label:
        font = ImageFont.truetype("Arial.ttf", size=max(round(max(im.size) / 40), 12))
        txt_width, txt_height = font.getsize(label)
        c = (box[0::2].mean(), box.ravel()[1::2].mean())
        draw.rectangle([c[0], c[1] - txt_height + 4, c[0] + txt_width, c[1]], fill=color)
        draw.text((c[0], c[1] - txt_height + 1), label, fill=(255, 255, 255), font=font)
    return np.asarray(im)


def polygon_output_to_target(output):
    # Convert model output to target format [batch_id, class_id, x1, y1, x2, y2, x3, y3, x4, y4, conf]
    targets = []
    for i, o in enumerate(output):
        for *box, conf, cls in o.cpu().numpy():
            targets.append([i, cls, *list(box), conf])
    return np.array(targets)


def polygon_plot_images(images, targets, paths=None, fname='images.jpg', names=None, max_size=640, max_subplots=16):
    # Plot image grid with labels
    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    # un-normalise
    if np.max(images[0]) <= 1:
        images *= 255

    tl = 3  # line thickness
    tf = max(tl - 1, 1)  # font thickness
    bs, _, h, w = images.shape  # batch size, _, height, width
    bs = min(bs, max_subplots)  # limit plot images
    ns = np.ceil(bs ** 0.5)  # number of subplots (square)

    # Check if we should resize
    scale_factor = max_size / max(h, w)
    if scale_factor < 1:
        h = math.ceil(scale_factor * h)
        w = math.ceil(scale_factor * w)

    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # init
    for i, img in enumerate(images):
        if i == max_subplots:  # if last batch has fewer images than we expect
            break

        block_x = int(w * (i // ns))
        block_y = int(h * (i % ns))

        img = img.transpose(1, 2, 0)
        if scale_factor < 1:
            img = cv2.resize(img, (w, h))

        mosaic[block_y:block_y + h, block_x:block_x + w, :] = img
        if len(targets) > 0:
            image_targets = targets[targets[:, 0] == i]
            boxes = image_targets[:, 2:10].T
            classes = image_targets[:, 1].astype('int')
            labels = image_targets.shape[1] == 10  # labels if no conf column
            conf = None if labels else image_targets[:, 10]  # check for confidence presence (label vs pred)

            if boxes.shape[1]:
                if boxes.max() <= 2.:  # if normalized with tolerance 1.0 (large tolerance allows four corners to be out of images)
                    boxes[0::2] *= w  # scale to pixels
                    boxes[1::2] *= h
                elif scale_factor < 1:  # absolute coords need scale if image scales
                    boxes *= scale_factor
            boxes[0::2] += block_x
            boxes[1::2] += block_y
            for j, box in enumerate(boxes.T):
                cls = int(classes[j])
                color = colors(cls)
                cls = names[cls] if names else cls
                if labels or conf[j] > 0.25:  # 0.25 conf thresh
                    label = '%s' % cls if labels else '%s %.1f' % (cls, conf[j])
                    polygon_plot_one_box(box, mosaic, label=label, color=color, line_thickness=tl)

        # Draw image filename labels
        if paths:
            label = Path(paths[i]).name[:40]  # trim to 40 char
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            cv2.putText(mosaic, label, (block_x + 5, block_y + t_size[1] + 5), 0, tl / 3, [220, 220, 220], thickness=tf,
                        lineType=cv2.LINE_AA)

        # Image border
        cv2.rectangle(mosaic, (block_x, block_y), (block_x + w, block_y + h), (255, 255, 255), thickness=3)

    if fname:
        r = min(1280. / max(h, w) / ns, 1.0)  # ratio to limit image size
        mosaic = cv2.resize(mosaic, (int(ns * w * r), int(ns * h * r)), interpolation=cv2.INTER_AREA)
        # cv2.imwrite(fname, cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB))  # cv2 save
        Image.fromarray(mosaic).save(fname)  # PIL save
    return mosaic


def polygon_plot_test_txt():  # from utils.plots import *; polygon_plot_test_txt()
    # Plot test.txt histograms
    x = np.loadtxt('test.txt', dtype=np.float32)
    box = x[:, :8]    #xyxyxyxy
    # using center points instead
    cx, cy = box[:, 0::2].mean(axis=1), box[:, 1::2].mean(axis=1)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)
    ax.hist2d(cx, cy, bins=600, cmax=10, cmin=0)
    ax.set_aspect('equal')
    plt.savefig('hist2d.png', dpi=300)
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)
    ax[0].hist(cx, bins=600)
    ax[1].hist(cy, bins=600)
    plt.savefig('hist1d.png', dpi=200)


def polygon_plot_targets_txt():  # from utils.plots import *; polygon_plot_targets_txt()
    # Plot targets.txt histograms
    x = np.loadtxt('targets.txt', dtype=np.float32).T
    s = ['x1 targets', 'y1 targets', 'x2 targets', 'y2 targets',
        'x3 targets', 'y3 targets', 'x4 targets', 'y4 targets']
    fig, ax = plt.subplots(4, 2, figsize=(16, 8), tight_layout=True)
    ax = ax.ravel()
    for i in range(8):
        ax[i].hist(x[i], bins=100, label='%.3g +/- %.3g' % (x[i].mean(), x[i].std()))
        ax[i].legend()
        ax[i].set_title(s[i])
    plt.savefig('targets.jpg', dpi=200)
    

def polygon_plot_labels(labels, names=(), save_dir=Path(''), loggers=None):
    # plot dataset labels
    LOGGER.info(f"Plotting labels to {save_dir / 'labels.jpg'}... ")
    c, b = labels[:, 0], labels[:, 1:].transpose()  # classes, boxes
    nc = int(c.max() + 1)  # number of classes
    x = pd.DataFrame(b.transpose(), columns=['x1', 'y1', 'x2', 'y2',
                                             'x3', 'y3', 'x4', 'y4'])

    # seaborn correlogram
    sn.pairplot(x, corner=True, diag_kind='auto', kind='hist', diag_kws=dict(bins=50), plot_kws=dict(pmax=0.9))
    plt.savefig(save_dir / 'labels_correlogram.png', dpi=200)
    plt.close()

    # matplotlib labels
    matplotlib.use('svg')  # faster
    ax = plt.subplots(3, 2, figsize=(12, 8), tight_layout=True)[1].ravel()
    y = ax[0].hist(c, bins=np.linspace(0, nc, nc + 1) - 0.5, rwidth=0.8)
    # [y[2].patches[i].set_color([x / 255 for x in colors(i)]) for i in range(nc)]  # update colors bug #3195 
    ax[0].set_ylabel('instances')
    if 0 < len(names) < 30:
        ax[0].set_xticks(range(len(names)))
        ax[0].set_xticklabels(names, rotation=90, fontsize=10)
    else:
        ax[0].set_xlabel('classes')
    for i in range(2, 6):
        sn.histplot(x, x=x.columns[2*(i-2)], y=x.columns[2*(i-2)+1], ax=ax[i], bins=50, pmax=0.9)

    # rectangles
    labels[:, 1:] *= 2000
    img = Image.fromarray(np.ones((2000, 2000, 3), dtype=np.uint8) * 255)
    for cls, *box in labels[:1000]:
        ImageDraw.Draw(img).polygon(box, outline=colors(cls))  # plot
    ax[1].imshow(img)
    ax[1].axis('off')

    for a in range(6):
        for s in ['top', 'right', 'left', 'bottom']:
            ax[a].spines[s].set_visible(False)

    plt.savefig(save_dir / 'labels.png', dpi=200)
    matplotlib.use('Agg')
    plt.close()


def plot_study_txt(path='', x=None):  # from utils.plots import *; plot_study_txt()
    # Plot study.txt generated by test.py
    plot2 = False  # plot additional results
    if plot2:
        ax = plt.subplots(2, 4, figsize=(10, 6), tight_layout=True)[1].ravel()

    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 4), tight_layout=True)
    # for f in [Path(path) / f'study_coco_{x}.txt' for x in ['yolov5s6', 'yolov5m6', 'yolov5l6', 'yolov5x6']]:
    for f in sorted(Path(path).glob('study*.txt')):
        y = np.loadtxt(f, dtype=np.float32, usecols=[0, 1, 2, 3, 7, 8, 9], ndmin=2).T
        x = np.arange(y.shape[1]) if x is None else np.array(x)
        if plot2:
            s = ['P', 'R', 'mAP@.5', 'mAP@.5:.95', 't_preprocess (ms/img)', 't_inference (ms/img)', 't_NMS (ms/img)']
            for i in range(7):
                ax[i].plot(x, y[i], '.-', linewidth=2, markersize=8)
                ax[i].set_title(s[i])

        j = y[3].argmax() + 1
        ax2.plot(y[5, 1:j], y[3, 1:j] * 1E2, '.-', linewidth=2, markersize=8,
                 label=f.stem.replace('study_coco_', '').replace('yolo', 'YOLO'))

    ax2.plot(1E3 / np.array([209, 140, 97, 58, 35, 18]), [34.6, 40.5, 43.0, 47.5, 49.7, 51.5],
             'k.-', linewidth=2, markersize=8, alpha=.25, label='EfficientDet')

    ax2.grid(alpha=0.2)
    ax2.set_yticks(np.arange(20, 60, 5))
    ax2.set_xlim(0, 57)
    ax2.set_ylim(30, 55)
    ax2.set_xlabel('GPU Speed (ms/img)')
    ax2.set_ylabel('COCO AP val')
    ax2.legend(loc='lower right')
    plt.savefig(str(Path(path).name) + '.png', dpi=300)


def plot_evolution(yaml_file='data/hyp.finetune.yaml'):  # from utils.plots import *; plot_evolution()
    # Plot hyperparameter evolution results in evolve.txt
    with open(yaml_file) as f:
        hyp = yaml.safe_load(f)
    x = np.loadtxt('evolve.txt', ndmin=2)
    f = fitness(x)
    # weights = (f - f.min()) ** 2  # for weighted results
    plt.figure(figsize=(10, 12), tight_layout=True)
    matplotlib.rc('font', **{'size': 8})
    for i, (k, v) in enumerate(hyp.items()):
        y = x[:, i + 7]
        # mu = (y * weights).sum() / weights.sum()  # best weighted result
        mu = y[f.argmax()]  # best single result
        plt.subplot(6, 5, i + 1)
        plt.scatter(y, f, c=hist2d(y, f, 20), cmap='viridis', alpha=.8, edgecolors='none')
        plt.plot(mu, f.max(), 'k+', markersize=15)
        plt.title('%s = %.3g' % (k, mu), fontdict={'size': 9})  # limit to 40 characters
        if i % 5 != 0:
            plt.yticks([])
        print('%15s: %.3g' % (k, mu))
    plt.savefig('evolve.png', dpi=200)
    print('\nPlot saved as evolve.png')


def polygon_plot_results(file=''):
    # Plot training 'results*.txt'. from utils.plots import *; plot_results(save_dir='runs/train/exp')
    fig, ax = plt.subplots(2, 5, figsize=(12, 6), tight_layout=True)
    ax = ax.ravel()
    s = ['Box', 'Objectness', 'Classification', 'Precision', 'Recall',
         'val Box', 'val Objectness', 'val Classification', 'mAP@0.5', 'mAP@0.5:0.95']
    start=0
    stop=0
    bucket=''
    id=()
    labels=()
    files = list(Path(file).glob('results*.csv'))
    assert len(files), f'No results.txt files found in {file.resolve()}, nothing to plot.'
    for fi, f in enumerate(files):
        try:
            results = np.loadtxt(f, usecols=[2, 3, 4, 8, 9, 12, 13, 14, 10, 11], ndmin=2).T
            n = results.shape[1]  # number of rows
            x = range(start, min(stop, n) if stop else n)
            for i in range(10):
                y = results[i, x]
                if i in [0, 1, 2, 5, 6, 7]:
                    y[y == 0] = np.nan  # don't show zero loss values
                    # y /= y[0]  # normalize
                label = labels[fi] if len(labels) else f.stem
                ax[i].plot(x, y, marker='.', label=label, linewidth=2, markersize=8)
                ax[i].set_title(s[i])
                # if i in [5, 6, 7]:  # share train and val loss y axes
                #     ax[i].get_shared_y_axes().join(ax[i], ax[i - 5])
        except Exception as e:
            print('Warning: Plotting error for %s; %s' % (f, e))

    ax[1].legend()
    fig.savefig(Path(file) / 'results.png', dpi=200)


def plot_one_box(x, im, color=(128, 128, 128), label=None, line_thickness=3):
    # Plots one bounding box on image 'im' using OpenCV
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)















