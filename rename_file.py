# encoding='UTF-8'
# author: pureyang
# TIME: 2019/8/22 下午5:15
# Description:数据图片的重命名

import os


def deal(path):
    file_names = os.listdir(path)
    c = 0
    # 随机获取一张图片的格式
    f_first = file_names[0]

    suffix = f_first.split('.')[-1]  # 图片文件的后缀
    for file in file_names:
        os.rename(os.path.join(path, file), os.path.join(path, '{:0>6d}.{}'.format(c, suffix)))
        c += 1


if __name__ == '__main__':
    deal('./data')  # 请按需修改图片文件的路径
