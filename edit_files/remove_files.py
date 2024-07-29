import os

# مسیر دایرکتوری که می خواهید فایل ها را در آن بررسی کنید را مشخص کنید


# تعداد مجاز فایل ها را تعیین کنید
max_files = 11

# تعداد فعلی فایل ها را در دایرکتوری شمارش کنید
for n in range(1, 10):
    directory = f"./dataset/main/image/sample{n}"
    num_files = len(os.listdir(directory))

    # اگر تعداد فایل ها بیشتر از حد مجاز باشد، فایل های اضافی را حذف کنید
    if num_files > max_files:
        for filename in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, filename)):
                os.remove(os.path.join(directory, filename))
                num_files -= 1
                if num_files <= max_files:
                    break

    print(f"تعداد فایل های حذف شده: {num_files}")
