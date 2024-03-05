from uwimg import *
import os
from glob import glob
from pathlib import Path

# resizing: kion
def resize(im_names, newpath, folder, sub_folder):
    for im_name in im_names:
        im = load_image(im_name)
        im = bilinear_resize(im, 128, 128)
        save_image(im, newpath + folder + sub_folder + Path(im_name).stem)


# converting to grayscale: bt
def convert_and_save(im_names, newpath, folder, sub_folder):
    for im_name in im_names:
        im = load_image(im_name)
        gray = rgb_to_grayscale(im)
        save_image(gray, newpath + folder + sub_folder + Path(im_name).stem)

# noise filter: evelyn
# takes a while; only use when necessary
def filter_noise(im_names, newpath, folder, sub_folder):
    for im_name in im_names:
        im = load_image(im_name)

        print("noise "+im_name)
        im = bilinear_resize(im, (int)(im.w*40), (int)(im.h*40))
        # im = convolve_image(im, make_gaussian_filter(1), 0)

        i = 0
        while i < im.w*im.h*im.c:
            if im.data[i] > .3:
                im.data[i] = 1
            i = i+1

        im = bilinear_resize(im, 128, 128)
        save_image(im, newpath + folder + sub_folder + Path(im_name).stem)


path = "C:\\Users\\evely\\CSE 455\\455finproj\\src\\wheres-waldo\\Hey-Waldo\\"
newpath = "C:\\Users\\evely\\CSE 455\\455finproj\\processed_data\\"
sub_folders = ["notwaldo", "waldo"]
folders = ["64", "64-bw", "64-gray", "128", "128-bw", "128-gray", "256", "256-bw", "256-gray", "original-images"]
for folder in folders:
    folder += "\\"
    if folder == "original-images\\":
        im_names = glob(os.path.join(path + folder, "*.jpg"))
        convert_and_save(im_names, newpath, folder, sub_folder="")
    elif folder == "64\\" or folder == "128\\" or folder == "256\\":
        for sub_folder in sub_folders:
            im_names = glob(os.path.join(path + folder, "*.jpg"))
            convert_and_save(im_names, newpath, folder, sub_folder="")
            resize(im_names, newpath, folder, sub_folder + "\\")
    elif folder == "64-gray\\" or folder == "128-gray\\" or folder == "256-gray\\":
        for sub_folder in sub_folders:
            im_names = glob(os.path.join(path + folder + sub_folder, "*.jpg"))
            filter_noise(im_names, newpath, folder, sub_folder + "\\")
            resize(im_names, newpath, folder, sub_folder + "\\")
    else:
        for sub_folder in sub_folders:
            im_names = glob(os.path.join(path + folder + sub_folder, "*.jpg"))
            resize(im_names, newpath, folder, sub_folder + "\\")
