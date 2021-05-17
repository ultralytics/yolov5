import sly_train_globals as g
import input_project as input_project
import classes as training_classes
import splits as train_val_split
import architectures as model_architectures
import hyperparameters as hyperparameters
import monitoring as monitoring
import artifacts as artifacts


def init(data, state):
    input_project.init(data)
    training_classes.init(g.api, data, state, g.project_id, g.project_meta)
    train_val_split.init(g.project_info, g.project_meta, data, state)
    model_architectures.init(data, state)
    hyperparameters.init(state)
    monitoring.init(data, state)
    artifacts.init(data)
