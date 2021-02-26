import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
import yaml

from tqdm import tqdm
import torch

sys.path.append(str(Path(__file__).parent.parent.parent))  # add utils/ to path
from utils.general import colorstr, xywh2xyxy
from utils.datasets import img2label_paths
from utils.datasets import LoadImagesAndLabels

try:
    import wandb
except ImportError:
    wandb = None
    print(f"{colorstr('wandb: ')}Install Weights & Biases for YOLOv5 logging with 'pip install wandb' (recommended)")

WANDB_ARTIFACT_PREFIX = 'wandb-artifact://'


def remove_prefix(from_string, prefix):
    return from_string[len(prefix):]

def get_id_and_model_name(run_path):
    # It's more elegant to stick to 1 wandb.init call, but as useful config data is overwritten in the WandbLogger's wandb.init call 
    if run_path.startswith(WANDB_ARTIFACT_PREFIX):
        run_path = Path(remove_prefix(run_path, WANDB_ARTIFACT_PREFIX))
        run_id = run_path.stem
        model_artifact_name = WANDB_ARTIFACT_PREFIX + 'run_' + run_id + '_model'
        assert wandb, 'install wandb to resume wandb runs'
        run = wandb.init(id=run_id, resume='allow') # Resume wandb-artifact:// runs here| workaround for not overwriting wandb.config
        return run, model_artifact_name
    return None, None

class WandbLogger():
    def __init__(self, opt, name, run_id, data_dict, job_type='Training'):
        # Pre-training routine -- check for --resume and --upload_dataset
        self.job_type = job_type
        self.wandb, self.wandb_run = wandb, None 
        if self.job_type == 'Training':
            run, model_artifact_name = self.check_resume(opt)
            if run:
                opt.resume = model_artifact_name
                opt.save_period = run.config.save_period
            if opt.upload_dataset:
                data_dict = self.check_and_upload_dataset(opt, name, data_dict, job_type)
            if self.wandb and not self.wandb_run:
                opt.data_dict = data_dict
                self.wandb_run = wandb.init(config=opt, 
                                            resume="allow",
                                            project='YOLOv5' if opt.project == 'runs/train' else Path(opt.project).stem,
                                            name=name,
                                            job_type=job_type,
                                            id=run_id) if not wandb.run else wandb.run
                self.setup_training(opt, data_dict)
        if self.job_type == 'Dataset Creation':
            self.data_dict = self.check_and_upload_dataset(opt, name, data_dict, job_type)

    
    def check_resume(self, opt):
        if self.job_type == 'Training':
            if isinstance(opt.resume, str):
                return get_id_and_model_name(opt.resume)
        return None, None
        
    def check_and_upload_dataset(self, opt, name, data_dict, job_type):
        assert wandb, 'Install wandb to upload dataset'
        run = wandb.init(config=data_dict, 
                   project='YOLOv5' if opt.project == 'runs/train' else Path(opt.project).stem,
                   name=name,
                   job_type="Dataset Creation") if not wandb.run else wandb.run
        config_path = self.create_dataset_artifact(opt.data, 
                                            opt.single_cls,
                                            'YOLOv5' if opt.project == 'runs/train' else Path(opt.project).stem)
        wandb.finish() # Finish dataset creation run| ensures the dataset has uploaded completely before training starts
        print("Created dataset config file ", config_path)
        with open(config_path) as f:
            wandb_data_dict = yaml.load(f, Loader=yaml.SafeLoader)
        return wandb_data_dict
                
    def setup_training(self, opt, data_dict):
        self.log_dict = {}
        self.current_epoch = 0
        self.log_imgs = 16 
        if opt.bbox_interval == -1:
            opt.bbox_interval = (opt.epochs // 10) if opt.epochs > 10 else opt.epochs
        if opt.resume:
            modeldir, _ = self.download_model_artifact(opt.resume)
            if modeldir:
                self.weights = Path(modeldir) / "best.pt"
                opt.weights = str(self.weights)
            data_dict = self.wandb_run.config.data_dict # Advantage: Eliminates the need for config file to resume
                
        self.train_artifact_path, self.train_artifact = \
            self.download_dataset_artifact(data_dict.get('train'), opt.artifact_alias)
        self.val_artifact_path, self.val_artifact = \
            self.download_dataset_artifact(data_dict.get('val'), opt.artifact_alias)
        
        self.result_artifact, self.result_table, self.weights = None, None, None
        if self.train_artifact_path is not None:
            train_path = Path(self.train_artifact_path) / 'data/images/'
            data_dict['train'] = str(train_path)
        if self.val_artifact_path is not None:
            val_path = Path(self.val_artifact_path) / 'data/images/'
            data_dict['val'] = str(val_path)
            self.result_artifact = wandb.Artifact("run_" + wandb.run.id + "_progress", "evaluation")
            self.result_table = wandb.Table(["epoch", "id", "prediction", "avg_confidence"])
            

    def download_dataset_artifact(self, path, alias):
        if path.startswith(WANDB_ARTIFACT_PREFIX):
            dataset_artifact = wandb.use_artifact(remove_prefix(path, WANDB_ARTIFACT_PREFIX) + ":" + alias)
            assert dataset_artifact is not None, "'Error: W&B dataset artifact doesn\'t exist'"
            datadir = dataset_artifact.download()
            labels_zip = Path(datadir) / "data/labels.zip"
            shutil.unpack_archive(labels_zip, Path(datadir) / 'data/labels', 'zip') if labels_zip.exists() else None
            return datadir, dataset_artifact
        return None, None

    def download_model_artifact(self, name):
        if name.startswith(WANDB_ARTIFACT_PREFIX):
            model_artifact = wandb.use_artifact(remove_prefix(name, WANDB_ARTIFACT_PREFIX)  + ":latest")
            assert model_artifact is not None, 'Error: W&B model artifact doesn\'t exist'
            modeldir = model_artifact.download()
            epochs_trained = model_artifact.metadata.get('epochs_trained')
            total_epochs = model_artifact.metadata.get('total_epochs')
            assert epochs_trained < total_epochs,  'training to %g epochs is finished, nothing to resume.' % (total_epochs)
            return modeldir, model_artifact
        return None, None

    def log_model(self, path, opt, epoch):
        datetime_suffix = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
        model_artifact = wandb.Artifact('run_' + wandb.run.id + '_model', type='model', metadata={
            'original_url': str(path),
            'epochs_trained': epoch + 1,
            'save period': opt.save_period,
            'project': opt.project,
            'datetime': datetime_suffix,
            'total_epochs': opt.epochs
        })
        model_artifact.add_file(str(path / 'last.pt'), name='last.pt')
        model_artifact.add_file(str(path / 'best.pt'), name='best.pt')
        wandb.log_artifact(model_artifact)
        print("Saving model artifact on epoch ", epoch + 1)

    def create_dataset_artifact(self, data_file, single_cls, project, overwrite_config=False):
        with open(data_file) as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)  # data dict 
        nc, names = (1, ['item']) if single_cls else (int(data['nc']), data['names'])
        names = {k: v for k, v in enumerate(names)}  # to index dictionary
        self.train_artifact = self.log_dataset_artifact(LoadImagesAndLabels(data['train']), names, name='train') if data.get('train') else None
        self.val_artifact = self.log_dataset_artifact(LoadImagesAndLabels(data['val']), names, name='val')  if data.get('val') else None            
        if data.get('train'):
            data['train'] = WANDB_ARTIFACT_PREFIX + str(Path(project) / 'train')
        if data.get('val'):
            data['val'] = WANDB_ARTIFACT_PREFIX + str(Path(project) / 'val')
        path = data_file if overwrite_config else data_file.replace('.', '_wandb.')  # updated data.yaml path
        data.pop('download', None)  # download via artifact instead of predefined field 'download:'
        with open(path, 'w') as f:
            yaml.dump(data, f)
        print("New Config file => ", path)
        return path
    
    def log_dataset_artifact(self, dataset, class_to_id, name='dataset'):
        # TODO: Explore multiprocessing to slpit this loop parallely| This is essential for speeding up the the logging
        # Process. We might need support for sorting wandb.Table object as if table is populated parallely.
        artifact = wandb.Artifact(name=name, type="dataset")
        for img_file in [dataset.path] if Path(dataset.path).is_dir() else dataset.img_files:
            if Path(img_file).is_dir():
                artifact.add_dir(img_file, name='data/images')
            else:
                artifact.add_file(img_file, name='data/images/'+Path(img_file).name)
                label_file = Path(img2label_paths([img_file])[0])
                artifact.add_file(str(label_file), name='data/labels/'+label_file.name) if label_file.exists() else None
        table = wandb.Table(columns=["id", "train_image", "Classes"])
        class_set = wandb.Classes([{'id': id, 'name': name} for id, name in class_to_id.items()])
        for si, (img, labels, paths, shapes) in enumerate(tqdm(dataset)):
            height, width = shapes[0]
            labels[:, 2:] = (xywh2xyxy(labels[:, 2:].view(-1, 4))) * torch.Tensor([width, height, width, height])
            box_data, img_classes = [], {}
            for cls, *xyxy in labels[:, 1:].tolist():
                cls = int(cls)
                box_data.append({"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
                                 "class_id": cls,
                                 "box_caption": "%s" % (class_to_id[cls]),
                                 "scores": {"acc": 1},
                                 "domain": "pixel"})
                img_classes[cls] = class_to_id[cls]
            boxes = {"ground_truth": {"box_data": box_data, "class_labels": class_to_id}}  # inference-space
            table.add_data(si, wandb.Image(paths, classes=class_set, boxes=boxes), json.dumps(img_classes))
        artifact.add(table, name)
        if Path(dataset.path).is_dir():
            labels_path = 'labels'.join(dataset.path.rsplit('images', 1))
            zip_path = Path(labels_path).parent / (name + '_labels.zip')
            if not zip_path.is_file():  # make_archive won't check if file exists
                shutil.make_archive(zip_path.with_suffix(''), 'zip', labels_path)
            artifact.add_file(str(zip_path), name='data/labels.zip')
        wandb.log_artifact(artifact)
        return artifact

    def log(self, log_dict):
        if self.wandb_run:
            for key, value in log_dict.items():
                self.log_dict[key] = value

    def end_epoch(self, best_result=False):
        if self.wandb_run and self.log_dict:
            wandb.log(self.log_dict)
            self.log_dict = {}
        if self.result_artifact:
            train_results = wandb.JoinedTable(self.val_artifact.get("val"), self.result_table, "id")
            self.result_artifact.add(train_results, 'result')
            wandb.log_artifact(self.result_artifact, aliases=['epoch '+ str(self.current_epoch),'best' if best_result else ''])
            self.result_table = wandb.Table(["epoch", "id", "prediction", "avg_confidence"])
            self.result_artifact = wandb.Artifact("run_" + wandb.run.id + "_progress", "evaluation")

    def finish_run(self):
        if self.wandb_run:
            if self.log_dict:
                wandb.log(self.log_dict)
            wandb.run.finish()
