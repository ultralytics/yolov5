python3 -m pip install --upgrade build
mkdir yolov5
cp -r data models models_v5.0 utils .pre-commit-config.yaml $(ls *.py) yolov5/
cat > yolov5/__init__.py << EOF
from export import run as export
from train import run as train
from val import run as val
EOF
python3 -m build
rm -r yolov5