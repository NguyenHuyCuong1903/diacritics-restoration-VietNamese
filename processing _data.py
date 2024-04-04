import re
import os
from tqdm import tqdm
from underthesea import sent_tokenize

def remove_tone_line(utf8_str):
    intab_l = "ạảãàáâậầấẩẫăắằặẳẵóòọõỏôộổỗồốơờớợởỡéèẻẹẽêếềệểễúùụủũưựữửừứíìịỉĩýỳỷỵỹđ"
    intab_u = "ẠẢÃÀÁÂẬẦẤẨẪĂẮẰẶẲẴÓÒỌÕỎÔỘỔỖỒỐƠỜỚỢỞỠÉÈẺẸẼÊẾỀỆỂỄÚÙỤỦŨƯỰỮỬỪỨÍÌỊỈĨÝỲỶỴỸĐ"
    intab = list(intab_l+intab_u)

    outtab_l = "a"*17 + "o"*17 + "e"*11 + "u"*11 + "i"*5 + "y"*5 + "d"
    outtab_u = "A"*17 + "O"*17 + "E"*11 + "U"*11 + "I"*5 + "Y"*5 + "D"
    outtab = outtab_l + outtab_u
    # Khởi tạo regex tìm kiếm các vị trí nguyên âm có dấu 'ạ|ả|ã|...'
    r = re.compile("|".join(intab))

    # Dictionary có key-value là từ có dấu-từ không dấu. VD: {'â' : 'a'}
    replaces_dict = dict(zip(intab, outtab))
    # Thay thế các từ có dấu xuất hiện trong tìm kiếm của regex bằng từ không dấu tương ứng
    non_dia_str = r.sub(lambda m: replaces_dict[m.group(0)], utf8_str)
    return non_dia_str

# Hàm chuẩn hoá câu
def standardize_data(row):
    # Xóa khoảng trắng dư thừa
    row = re.sub(r'\s+', " ", row)
    return row

with open('./Data/data.txt', 'r', encoding='utf-8') as f:
    data = f.readlines()
    f.close()
    X_data, y_data = [], []
    for i in range(len(data)):
        for it in sent_tokenize(data[i]):
            it = standardize_data(it)
            if 10 < len(it.split()) < 100:  #  Xóa bỏ câu ít hơn 6 từ và dài hơn 100
                y_data.append(f"{it}\n")
                X_data.append(f"{remove_tone_line(it)}\n")

import random

# Combine X and y data
combined_data = list(zip(X_data, y_data))
# Shuffle the combined data
random.shuffle(combined_data)

# Determine the sizes of each split
total_samples = len(combined_data)
train_size = int(0.6 * total_samples)
test_size = int(0.2 * total_samples)
valid_size = total_samples - train_size - test_size

# Split the data
train_data = combined_data[:train_size]
test_data = combined_data[train_size:train_size + test_size]
valid_data = combined_data[train_size + test_size:]

print("Train size: ", train_size)
print("Test size: ", test_size)
print("Valid size: ", valid_size)

# Write data to separate files for each split
def write_data(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as file:
        for item in data:
            file.write("".join(item))

write_data('./Data/train.txt', [item[0] for item in train_data])
write_data('./Data/train_Label.txt', [item[1] for item in train_data])

write_data('./Data/test.txt', [item[0] for item in test_data])
write_data('./Data/test_Label.txt', [item[1] for item in test_data])

write_data('./Data/valid.txt', [item[0] for item in valid_data])
write_data('./Data/valid_Label.txt', [item[1] for item in valid_data])

print("Data splitting completed.")
