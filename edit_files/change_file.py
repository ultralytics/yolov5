import os
from PIL import Image

def convert_images(path):
  """
  فایل های JPEG موجود در پوشه `path` را به PNG تبدیل می کند.
  """
  for file in os.listdir(path):
    if file.endswith(".jpg") or file.endswith(".jpeg"):
      # نام فایل را بدون پسوند دریافت کنید
      base_name, ext = os.path.splitext(file)
      # تصویر را به عنوان JPEG بارگیری کنید
      image = Image.open(os.path.join(path, file))
      # تصویر را به فرمت PNG تبدیل کنید
      rgb_image = image.convert('RGB')
      # تصویر PNG را با نام جدید ذخیره کنید
      rgb_image.save(os.path.join(path, base_name + '.png'))

# مثال استفاده
path = "./dataset/main/all_files_together"
convert_images(path)