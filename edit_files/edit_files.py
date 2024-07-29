import os

from PIL import Image

# مسیر پوشه مقصد را وارد کنید
folder_image_path = "./dataset/main/all_files_together"

# برای هر فایل در پوشه:
for filename in os.listdir(folder_image_path):
    if filename.endswith(".jpeg"):
        # نام فعلی فایل را دریافت کنید
        current_name = filename
        # مسیر کامل فایل JPEG
        current_path = os.path.join(folder_image_path, current_name)

        # باز کردن فایل JPEG
        with Image.open(current_path) as img:
            # نام جدید فایل PNG
            new_name = current_name[:-5] + ".png"
            new_path = os.path.join(folder_image_path, new_name)
            # ذخیره فایل به عنوان PNG
            img.save(new_path, "PNG")

        # حذف فایل JPEG اصلی
        os.remove(current_path)
