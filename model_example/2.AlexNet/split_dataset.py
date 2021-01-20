# dogs vs cats dataset
# https://www.kaggle.com/c/dogs-vs-cats/data
import os, shutil

dataset_path = './datasets/dogs_vs_cats/train'
output_dir = './datasets/dogs_vs_cats_split'
path_dir = {}
split_dataset_folder = ['train', 'validation', 'test']
labels = ['cats', 'dogs']
split_num = {
    'train':[0, 7500], 
    'validation':[7500, 10000],
    'test':[10000, 12500],
}

if os.path.exists(output_dir):  # 반복적인 실행을 위해 디렉토리를 삭제합니다.
    shutil.rmtree(output_dir)   # 이 코드는 책에 포함되어 있지 않습니다.
os.mkdir(output_dir)

for dir_name in split_dataset_folder:
    path_dir[dir_name] = os.path.join(output_dir, dir_name)
    os.mkdir(path_dir[dir_name])
    for label in labels:
        path_dir[dir_name + '/' + label] = os.path.join(path_dir[dir_name], label)
        os.mkdir(path_dir[dir_name + '/' + label])

def move_dataset(label, start, end, from_dir, to_dir):
    fnames = ['{}.{}.jpg'.format(label[:-1], i) for i in range(start, end)]
    for fname in fnames:
        src = os.path.join(from_dir, fname)
        dst = os.path.join(to_dir, fname)
        shutil.copyfile(src, dst)

for dir_name in split_dataset_folder:
    for label in labels:
        path_name = os.path.join(dir_name, label)
        move_dataset(label, split_num[dir_name][0], split_num[dir_name][1], dataset_path, path_dir[path_name])
        print('{} {} image count: {}'.format(dir_name, label, len(os.listdir(path_dir[path_name]))))