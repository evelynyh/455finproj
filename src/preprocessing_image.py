from uwimg import *
import os
from glob import glob

# by Wednesday, preprocessing

# resizing: kion
  #l1 normalize wherever
  # 128x128
  # bilinear interpolate

# converting to grayscale: bt
  # l1 normalize wherever


def convert_and_save(im_names, newpath, folder, sub_folder):
    for im_name in im_names:
        im = load_image(im_name)
        gray = rgb_to_grayscale(im)
        save_image(gray, newpath + folder + "\\" + sub_folder + os.path.basename(im_name))

# noise filter: evelyn
def filter_noise(img_set, newpath, folder, sub_folder):
    for x in img_set:
        im = load_image(x)
        # Opt 1: shrink grow then highpass
        im = bilinear_resize(im, im.w/4, im.h/4)
        im = bilinear_resize(im, im.w*4, im.h*4)

        # Opt 2: use gauss (please test)
        # im = convolve_image(im, make_gaussian_filter(2), 0)

        new_im = convolve_image(im, make_highpass_filter(), 0)

        save_image(new_im, newpath + folder + "\\" + sub_folder + os.path.basename(x))



path = "C:\\Users\\evely\\CSE 455\\455finproj\\wheres-waldo\\Hey-Waldo\\"
newpath = "C:\\Users\\evely\\CSE 455\\455finproj\\processed_data\\"
folders = ["64", "128", "256", "original-images"]
sub_folders = ["notwaldo", "waldo"]
for folder in folders:
    if folder == "original-images":
        im_names = glob(os.path.join(path + folder + "\\", "*.jpg"))

        convert_and_save(im_names, newpath, folder, sub_folder="")

    else:
        for sub_folder in sub_folders:
            im_names = glob(os.path.join(path + folder + "\\" + sub_folder, "*.jpg"))
            convert_and_save(im_names, newpath, folder, sub_folder + "\\")

folders_gray = ["64-gray", "128-gray", "256-gray"]
for folder in folders_gray:
    for sub_folder in sub_folders:
        im_names = glob(os.path.join(path + folder + "\\" + sub_folder, "*.jpg"))
        filter_noise(im_names, newpath, folder, sub_folder + "\\")
