python3 -m pip install --upgrade build
mkdir yolov5
git clone --depth=1 . yolov5/
rm -rf yolov5/.git
grep --include=\*.py -rnl 'yolov5/' -e "from models" | xargs -i@ sed -i 's/from models/from yolov5.models/g' @
grep --include=\*.py -rnl 'yolov5/' -e "from utils" | xargs -i@ sed -i 's/from utils/from yolov5.utils/g' @
grep --include=\*.py -rnl 'yolov5/' -e "from train" | xargs -i@ sed -i 's/from train/from yolov5.train/g' @
grep --include=\*.py -rnl 'yolov5/' -e "from val" | xargs -i@ sed -i 's/from val/from yolov5.val/g' @
grep --include=\*.py -rnl 'yolov5/' -e "from export" | xargs -i@ sed -i 's/from export/from yolov5.export/g' @
grep --include=\*.py -rnl 'yolov5/' -e "from classify" | xargs -i@ sed -i 's/from classify/from yolov5.classify/g' @
grep --include=\*.py -rnl 'yolov5/' -e "from segment" | xargs -i@ sed -i 's/from segment/from yolov5.segment/g' @
sed -i '/^sparseml/d' yolov5/requirements.txt
python3 -m build
rm -r yolov5