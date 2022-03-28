import numpy as np
import os

module_name = ["2yel"]

def transform_single_frame(path,module_name,file_name):
    fin = open(path+file_name, "rt")
    fout = open("out.txt", "wt")

    for line in fin:
        if line.endswith('e\n'):
            line = line[:-1] + "+00"
            print(' '.join(line.split()))
        fout.write(' '.join(line.split()))
        fout.write("\n")

    fin.close()
    fout.close()

    npdata = np.loadtxt('out.txt',delimiter=" ")

    npy_name = file_name.split(".")[0]+".npy"

    fout = open(f"data/b{module_name}/{npy_name}", "wb")
    np.save(fout, npdata)


def transform_trajactory(path, module_name):
    print(path)
    files = os.listdir(path) #采用listdir来读取所有文件
    files.sort() #排序
    print(f"transforming {module_name}...")
    for file_ in files:     #循环读取每个文件名
        f_name = str(file_)
        print(f"transforming {f_name}...")
        transform_single_frame(path,module_name,f_name)


def traverse_directory(path):
    files = os.listdir(path)
    for file_ in files:
        if os.path.isdir(path+file_):
            if file_ in module_name:
                transform_trajactory(path+file_+"/", file_)

traverse_directory("data/")