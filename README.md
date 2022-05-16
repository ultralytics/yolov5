# MOPS: Scrap Yard Managment

This is a forked repository from YOLOv5 customized to be used for scrap yard images.

For full documentation on training, testing and deployment, please see the [YOLOv5 Docs](https://docs.ultralytics.com).

## <div align="center">Quick Start Examples</div>

<details open>
<summary>Install</summary>

Clone repo and install [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) in a
[**Python>=3.7.0**](https://www.python.org/) environment, including
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/).

```bash
git clone ssh://git@git.tech.sms-digital.cloud:2222/smsdigital/ppq_sym_ml.git  # clone
cd ppq_sym_ml
pip install -r requirements.txt  # install
```

#### Data retrieval using DVC
The datasets used in  this project can be retrieved and modified using [**DVC**](https://dvc.org/) as its data versioning tool.

#### Setting up dvc authentication
To be able to pull from and push to the dvc remote an AWS account with access to
SBX-DeepLearning-DEV-20211018 is
required. Please ask the Infrastructure team for an account with the following role:
`SBX-DeepLearning-DEV-20211018 AWSPowerUserAccess`.

The AWS CLI needs to be installed. The installation package and documentation can be found under
the [official AWS CLI documentation](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-install.html).

`AWS sso` is utilized for authentication. To set this up, run in your terminal: `aws configure`

Provide the `AWS Access Key ID` and `AWS Secret Access Key`, which  you can find here:
https://d-936713fd01.awsapps.com/start#/. Set the `Default region name` to `eu-central-1` and
the `Default output format` to `None`.

Next, you need to configure the AWS profile. Open `~/.aws/config` and append the following text:
```
[profile 260171390835_AWSPowerUserAccess]
sso_start_url = https://d-936713fd01.awsapps.com/start
sso_region = eu-west-1
sso_account_id = 260171390835
sso_role_name = AWSPowerUserAccess
region = eu-central-1
```

Finally, you need to provide the credentials, that you can find in the command line access under
the `AWSPowerUserAccess` profile at: https://d-936713fd01.awsapps.com/start#/. Open `~/.aws/credentials` and paste
the credentials there. If you have any questions feel free to reach out to your colleagues, they
can help you with the setup.

Before you are able to interact with AWS buckets, you will need to log in using AWS CLI:
```bash
aws sso login --profile 260171390835_AWSPowerUserAccess
```
You can now retrieve the dataset using:
```bash
dvc pull
```
And in case you need to make modifications to dataset, you can push your changes by doing:
```bash
dvc push
```

For more information on dvc and the installation procedure please check the dvc Confluence page" [dvc documentation](https://smsdigital.atlassian.net/wiki/spaces/DS/pages/2834661483/DVC+-+dataset+version+control)
</details>

<details open>
<summary>Inference with detect.py</summary>

`detect.py` runs inference on a variety of sources, downloading [models](https://github.com/ultralytics/yolov5/tree/master/models) automatically from
the latest YOLOv5 [release](https://github.com/ultralytics/yolov5/releases) and saving results to `runs/detect`.

```bash
python detect.py --source 0  # webcam
                          img.jpg  # image
                          vid.mp4  # video
                          path/  # directory
                          path/*.jpg  # glob
                          'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                          'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
```
</details>

<details open>
<summary>Training</summary>

The commands below reproduce YOLOv5 sample SYM results. Use the
largest `--batch-size` possible, or pass `--batch-size -1` for
YOLOv5 [AutoBatch](https://github.com/ultralytics/yolov5/pull/5092). Batch sizes shown for V100-16GB.
Please note that providing a name for the experience is necessary.

```bash
python train.py --name sample-exp --data sample-sym.yaml --cfg yolov5n.yaml --weights '' --batch-size 128
                                       yolov5s                                64
                                       yolov5m                                40
                                       yolov5l                                24
                                       yolov5x                                16
```

<img width="800" src="https://user-images.githubusercontent.com/26833433/90222759-949d8800-ddc1-11ea-9fa1-1c97eed2b963.png">
</details>


### DVC pipelines

Other than handling the data files, DVC can also be used to train, test and trigger mlflow ui.

#### Params
You can modify the default `params.yaml` file to point to your data with any specification you require.
Default values are:
```yaml
train_data: sym-sample.yaml
detection_source: data/datasets/sym/images/test
epochs : 1000
batch-size: 10
```

#### Training pipeline
In order to trigger the training you can use the command:
```bash
dvc repro train
```
#### Detecting pipeline
After the training is done, you can always run the following command to detect images from any source with the latest best weights.
```bash
dvc repro detect
```
#### MLflow
Finally, you can observe and track the outcome of your experiments using MLflow:
```bash
dvc repro mlflow
```
