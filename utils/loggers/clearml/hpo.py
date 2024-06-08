# Ultralytics YOLOv5 🚀, AGPL-3.0 license

from clearml import Task

# Connecting ClearML with the current process,
# from here on everything is logged automatically
from clearml.automation import HyperParameterOptimizer, UniformParameterRange
from clearml.automation.optuna import OptimizerOptuna

task = Task.init(
    project_name="Hyper-Parameter Optimization",
    task_name="YOLOv5",
    task_type=Task.TaskTypes.optimizer,
    reuse_last_task_id=False,
)

# Example use case:
optimizer = HyperParameterOptimizer(
    # This is the experiment we want to optimize
    base_task_id="<your_template_task_id>",
    # here we define the hyper-parameters to optimize
    # Notice: The parameter name should exactly match what you see in the UI: <section_name>/<parameter>
    # For Example, here we see in the base experiment a section Named: "General"
    # under it a parameter named "batch_size", this becomes "General/batch_size"
    # If you have `argparse` for example, then arguments will appear under the "Args" section,
    # and you should instead pass "Args/batch_size"
    hyper_parameters=[
        UniformParameterRange("Hyperparameters/lr0", min_value=1e-5, max_value=1e-1),
        UniformParameterRange("Hyperparameters/lrf", min_value=0.01, max_value=1.0),
        UniformParameterRange("Hyperparameters/momentum", min_value=0.6, max_value=0.98),
        UniformParameterRange("Hyperparameters/weight_decay", min_value=0.0, max_value=0.001),
        UniformParameterRange("Hyperparameters/warmup_epochs", min_value=0.0, max_value=5.0),
        UniformParameterRange("Hyperparameters/warmup_momentum", min_value=0.0, max_value=0.95),
        UniformParameterRange("Hyperparameters/warmup_bias_lr", min_value=0.0, max_value=0.2),
        UniformParameterRange("Hyperparameters/box", min_value=0.02, max_value=0.2),
        UniformParameterRange("Hyperparameters/cls", min_value=0.2, max_value=4.0),
        UniformParameterRange("Hyperparameters/cls_pw", min_value=0.5, max_value=2.0),
        UniformParameterRange("Hyperparameters/obj", min_value=0.2, max_value=4.0),
        UniformParameterRange("Hyperparameters/obj_pw", min_value=0.5, max_value=2.0),
        UniformParameterRange("Hyperparameters/iou_t", min_value=0.1, max_value=0.7),
        UniformParameterRange("Hyperparameters/anchor_t", min_value=2.0, max_value=8.0),
        UniformParameterRange("Hyperparameters/fl_gamma", min_value=0.0, max_value=4.0),
        UniformParameterRange("Hyperparameters/hsv_h", min_value=0.0, max_value=0.1),
        UniformParameterRange("Hyperparameters/hsv_s", min_value=0.0, max_value=0.9),
        UniformParameterRange("Hyperparameters/hsv_v", min_value=0.0, max_value=0.9),
        UniformParameterRange("Hyperparameters/degrees", min_value=0.0, max_value=45.0),
        UniformParameterRange("Hyperparameters/translate", min_value=0.0, max_value=0.9),
        UniformParameterRange("Hyperparameters/scale", min_value=0.0, max_value=0.9),
        UniformParameterRange("Hyperparameters/shear", min_value=0.0, max_value=10.0),
        UniformParameterRange("Hyperparameters/perspective", min_value=0.0, max_value=0.001),
        UniformParameterRange("Hyperparameters/flipud", min_value=0.0, max_value=1.0),
        UniformParameterRange("Hyperparameters/fliplr", min_value=0.0, max_value=1.0),
        UniformParameterRange("Hyperparameters/mosaic", min_value=0.0, max_value=1.0),
        UniformParameterRange("Hyperparameters/mixup", min_value=0.0, max_value=1.0),
        UniformParameterRange("Hyperparameters/copy_paste", min_value=0.0, max_value=1.0),
    ],
    # this is the objective metric we want to maximize/minimize
    objective_metric_title="metrics",
    objective_metric_series="mAP_0.5",
    # now we decide if we want to maximize it or minimize it (accuracy we maximize)
    objective_metric_sign="max",
    # let us limit the number of concurrent experiments,
    # this in turn will make sure we don't bombard the scheduler with experiments.
    # if we have an auto-scaler connected, this, by proxy, will limit the number of machine
    max_number_of_concurrent_tasks=1,
    # this is the optimizer class (actually doing the optimization)
    # Currently, we can choose from GridSearch, RandomSearch or OptimizerBOHB (Bayesian optimization Hyper-Band)
    optimizer_class=OptimizerOptuna,
    # If specified only the top K performing Tasks will be kept, the others will be automatically archived
    save_top_k_tasks_only=5,  # 5,
    compute_time_limit=None,
    total_max_jobs=20,
    min_iteration_per_job=None,
    max_iteration_per_job=None,
)

# report every 10 seconds, this is way too often, but we are testing here
optimizer.set_report_period(10 / 60)
# You can also use the line below instead to run all the optimizer tasks locally, without using queues or agent
# an_optimizer.start_locally(job_complete_callback=job_complete_callback)
# set the time limit for the optimization process (2 hours)
optimizer.set_time_limit(in_minutes=120.0)
# Start the optimization process in the local environment
optimizer.start_locally()
# wait until process is done (notice we are controlling the optimization process in the background)
optimizer.wait()
# make sure background optimization stopped
optimizer.stop()

print("We are done, good bye")
