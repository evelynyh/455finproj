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


path = "C:\\Users\\bt084\\455finproj\\wheres-waldo\\Hey-Waldo\\"
newpath = "C:\\Users\\bt084\\455finproj\\processed_data\\"
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


# noise filter: evelyn
  # edge supression: run filter
  # shrink-grow
  #l1 normalize if needed
