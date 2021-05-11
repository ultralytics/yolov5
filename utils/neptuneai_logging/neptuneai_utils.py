from pathlib import Path
from utils.general import colorstr

try:
    import neptune.new as neptune
except ImportError:
    neptune = None


class NeptuneLogger:
    def __init__(self, opt, name, data_dict, job_type='Training'):
        # Pre-training routine --
        self.job_type = job_type
        self.neptune, self.neptune_run, self.data_dict = neptune, None, data_dict

        if self.neptune and opt.neptune_token:
            self.neptune_run = neptune.init(api_token=opt.neptune_token,
                                            project=opt.neptune_project,
                                            name=name)
        if self.neptune_run:
            if self.job_type == 'Training':
                if not opt.resume:
                    neptune_data_dict = data_dict
                    self.neptune_run["opt"] = vars(opt)
                    self.neptune_run["data_dict"] = neptune_data_dict
                self.data_dict = self.setup_training(data_dict)
            prefix = colorstr('neptune: ')
            print(f"{prefix}NeptuneAI logging initiated successfully.")
        else:
            #prefix = colorstr('neptune: ')
            #print(
            #    f"{prefix}Install NeptuneAI for YOLOv5 logging with 'pip install neptune-client' (recommended)")
            pass

    def setup_training(self, data_dict):
        self.log_dict, self.current_epoch = {}, 0  # Logging Constants
        return data_dict

    def log(self, log_dict):
        if self.neptune_run:
            for key, value in log_dict.items():
                self.log_dict[key] = value

    def end_epoch(self, best_result=False):
        if self.neptune_run:
            for key, value in self.log_dict.items():
                self.neptune_run[key].log(value)
                self.log_dict = {}

    def finish_run(self):
        if self.neptune_run:
            if self.log_dict:
                for key, value in self.log_dict.items():
                    self.neptune_run[key].log(value)
            self.neptune_run.stop()
