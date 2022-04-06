python3 -m pip install --upgrade build
mkdir yolov5
cp -r data models models_v5.0 utils .pre-commit-config.yaml $(ls *.py) yolov5/
touch yolov5/__init__.py
python3 -m build