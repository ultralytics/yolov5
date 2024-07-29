import os
import shutil

# for n in range(1,10):

# input 2 folders
source_folders = ["./dataset/main/images", "./dataset/main/labels"]
# Destination
destination_folder = "./dataset/main/all_files_together"

# move every files
for folder in source_folders:
    # برای هر فایل در پوشه مبدا:
    for filename in os.listdir(folder):
        # مسیر کامل فایل را بدست آورید
        source_path = os.path.join(folder, filename)
        # مسیر کامل فایل در پوشه مقصد را بسازید
        dest_path = os.path.join(destination_folder, filename)
        # فایل را به پوشه مقصد جابجا کنید
        shutil.move(source_path, dest_path)

print("sample mission successed")
