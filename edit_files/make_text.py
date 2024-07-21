import os

n = 9
# مسیر دایرکتوری حاوی فایل های png
directory = "./dataset/main/images"
# مسیر ذخیره سازی فایل های txt (پوشه x)
txt_save_dir = "./dataset/main/labels"
if not os.path.exists(txt_save_dir):
    os.mkdir(txt_save_dir)  # پوشه x را اینجا تغییر دهید

# برای هر فایل png در دایرکتوری:
for filename in os.listdir(directory):
    # اگر فایل png بود:
    if filename.endswith(".png" or ".jpg"):
        # از نام فایل، نام فایل txt رو با جایگزینی png با txt بسازید
        txt_filename = os.path.join(txt_save_dir, filename[:-4] + ".txt")

        # فایل txt رو با حالت نوشتن باز کنید
        with open(txt_filename, "w") as txt_file:
            # عدد 1 رو داخل فایل txt بنویسید
            txt_file.write(f"{n-1} 0.51875 0.5046875 0.76953125 0.8421875")
