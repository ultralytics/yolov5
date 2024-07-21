import os

def remove_jpegs(path):
  """
  تمام فایل های JPEG موجود در پوشه `path` و زیر پوشه های آن را حذف می کند.
  """
  for file in os.walk(path):
    for f in file[2]:
      if f.endswith(".jpeg"):
        os.remove(os.path.join(file[0], f))

# مثال استفاده
path = "./dataset/main/all_files_together"
remove_jpegs(path)