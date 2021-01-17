from datetime import datetime
import shutil
from pathlib import Path
import os
import stat
import logging

import tqdm
import torch
import json
from utils.general import colorstr, xywh2xyxy
logger = logging.getLogger(__name__)

try:
    import wandb
except ImportError:
    wandb = None
    print(f"{colorstr('wandb: ')}Install Weights & Biases for YOLOv5 logging with 'pip install wandb' (recommended)")

WANDB_ARTIFACT_PREFIX = 'wandb-artifact://'

def remove_prefix(from_string, prefix):
    return from_string[len(prefix):]
    
class WandbLogger():
    def __init__(self, opt, name, run_id, data_dict, job_type='Training'):
        self.wandb = wandb
        if self.wandb:
            self.wandb_run = wandb.init(config=opt, resume="allow",
                                   project='YOLOv5' if opt.project == 'runs/train' else Path(opt.project).stem,
                                   name=name,
                                   job_type=job_type,
                                   id=run_id)
        else:
            self.wandb_run = None
        if job_type == 'Training':
            self.setup_training(opt, data_dict)
            if opt.bbox_interval == -1:
                opt.bbox_interval = (opt.epochs // 10) if opt.epochs > 10 else opt.epochs
            if opt.save_period == -1:
                opt.save_period = (opt.epochs // 10) if opt.epochs > 10 else opt.epochs
    
    def setup_training(self, opt, data_dict):
        self.log_dict = {}
        self.train_artifact_path, self.trainset_artifact = self.download_dataset_artifact(data_dict['train'], opt.artifact_alias)
        self.test_artifact_path, self.testset_artifact = self.download_dataset_artifact(data_dict['val'], opt.artifact_alias)
        self.result_artifact, self.result_table, self.weights = None, None, None
        if self.train_artifact_path is not None:
            train_path = self.train_artifact_path + '/data/images/'
            data_dict['train'] = train_path
        if self.test_artifact_path is not None:
            test_path = self.test_artifact_path + '/data/images/'
            data_dict['val'] = test_path
            self.result_artifact = wandb.Artifact("run_"+wandb.run.id+"_progress", "evaluation")
            self.result_table = wandb.Table(["epoch", "id", "prediction", "avg_confidence"])
        if opt.resume_from_artifact:
            modeldir, _ = self.download_model_artifact(opt.resume_from_artifact)
            if modeldir:
                self.weights = modeldir + "/best.pt"
                opt.weights = self.weights
        

    def download_dataset_artifact(self,path, alias):
        if path.startswith(WANDB_ARTIFACT_PREFIX):
            dataset_artifact = wandb.use_artifact(remove_prefix(path,WANDB_ARTIFACT_PREFIX)+":"+alias)
            if dataset_artifact is None:
                logger.error('Error: W&B dataset artifact doesn\'t exist')
                raise ValueError('Artifact doesn\'t exist')
            datadir = dataset_artifact.download()
            labels_zip = datadir+"/data/labels.zip"
            shutil.unpack_archive(labels_zip, datadir+'/data/labels', 'zip')
            print("Downloaded dataset to : ", datadir)
            return datadir, dataset_artifact
        return None, None
    
    def download_model_artifact(self,name):
        model_artifact = wandb.use_artifact(name+":latest")
        if model_artifact is None:
            logger.error('Error: W&B model artifact doesn\'t exist')
            raise ValueError('Artifact doesn\'t exist')
        modeldir = model_artifact.download()
        print("Downloaded model to : ", modeldir)
        return modeldir, model_artifact
    
    def log_model(self, path, opt, epoch):
        datetime_suffix = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
        model_artifact = wandb.Artifact('run_'+wandb.run.id+'_model', type='model', metadata={
            'original_url': str(path),
            'epoch': epoch+1,
            'save period': opt.save_period,
            'project': opt.project,
            'datetime': datetime_suffix
        })
        model_artifact.add_file(str(path / 'last.pt'), name='last.pt')
        model_artifact.add_file(str(path / 'best.pt'), name='best.pt')
        wandb.log_artifact(model_artifact)

        if epoch+1 == opt.epochs:
            model_artifact = wandb.Artifact('final_model', type='model', metadata={
            'run_id': wandb.run.id,
            'datetime': datetime_suffix
            })
            model_artifact.add_file(str(path / 'last.pt'), name='last.pt')
            model_artifact.add_file(str(path / 'best.pt'), name='best.pt')
            wandb.log_artifact(model_artifact)
        print("Saving model artifact on epoch ", epoch+1)
    
    def log_dataset_artifact(self, dataloader, device, class_to_id, name='dataset'):
        artifact = wandb.Artifact(name=name, type="dataset")
        image_path = dataloader.dataset.path
        artifact.add_dir(image_path,name='data/images')
        table = wandb.Table(
            columns=["id", "train_image", "Classes"]
        )
        id_count = 0
        class_set = wandb.Classes([{'id':id , 'name':name} for id,name in class_to_id.items()])
        for batch_i, (img, targets, paths, shapes) in enumerate(dataloader):
            targets = targets.to(device)
            nb, _, height, width = img.shape  # batch size, channels, height, width
            targets[:,2:] = (xywh2xyxy(targets[:,2:].view(-1, 4)))
            for si, _ in enumerate(img):
                height, width = shapes[si][0]
                labels = targets[targets[:, 0] == si]
                labels[:,2:] *=  torch.Tensor([width, height, width, height]).to(device)
                labels = labels[:, 1:]
                box_data = []
                img_classes = {}
                for cls, *xyxy in labels.tolist():
                  class_id = int(cls)
                  box_data.append({"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
                                  "class_id": class_id,
                                  "box_caption": "%s" % (class_to_id[class_id]),
                                  "scores": {"acc": 1},
                                  "domain": "pixel"})
                  img_classes[class_id] = class_to_id[class_id]
                boxes = {"ground_truth": {"box_data": box_data, "class_labels": class_to_id}}  # inference-space
                table.add_data(id_count,wandb.Image(paths[si], classes=class_set, boxes=boxes), json.dumps(img_classes))
                id_count = id_count+1
        artifact.add(table, name)
        label_path = image_path.replace('images','labels')
        # Workaround for: Unable to log empty txt files via artifacts
        if not os.path.isfile(name+'_labels.zip'): # make_archive won't check if file exists
            shutil.make_archive(name+'_labels', 'zip', label_path)
        artifact.add_file(name+'_labels.zip', name='data/labels.zip')
        wandb.log_artifact(artifact)
        print("Saving data to W&B...")
            
    def log(self, log_dict):
        if self.wandb_run:
            for key, value in log_dict.items():
                self.log_dict[key] = value
                
    def end_epoch(self):
        if self.wandb_run and self.log_dict:
            wandb.log(self.log_dict)
        self.log_dict = {}

    def finish_run(self):
        if self.wandb_run:
            if self.result_artifact:
                print("Add Training Progress Artifact")
                self.result_artifact.add(self.result_table, 'result')
                train_results = wandb.JoinedTable(self.testset_artifact.get("val"), self.result_table, "id")
                self.result_artifact.add(train_results, 'joined_result')
                wandb.log_artifact(self.result_artifact)
            if self.log_dict:
                wandb.log(self.log_dict)
            wandb.run.finish()
