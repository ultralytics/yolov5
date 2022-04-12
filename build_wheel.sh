python3 -m pip install --upgrade build
mkdir yolov5
cp -r data models models_v5.0 utils .pre-commit-config.yaml $(ls *.py) requirements.txt yolov5/
grep --include=\*.py -rnl 'yolov5/' -e "from models" | xargs -i@ sed -i 's/from models/from yolov5.models/g' @
grep --include=\*.py -rnl 'yolov5/' -e "from utils" | xargs -i@ sed -i 's/from utils/from yolov5.utils/g' @
sed -i '$d' yolov5/requirements.txt
cat > yolov5/__init__.py << EOF
EOF
python3 -m build
rm -r yolov5
