"""
Usage: img_test.py [-s] <PATH>
"""

import colorsys
import os
import math
from collections import defaultdict

from scipy.misc import imresize
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageStat
from docopt import docopt
from pylab import *
import matplotlib.pyplot as plt


def extract_demo(path, show):
    filename, ext = os.path.splitext(path)
    img = Image.open(path)
    width, height = img.size # width, height of image
    pixel_count = width*height

    img_gray = img.convert('L')
    hist = img_gray.histogram()
    hist_norm = []
    for g in hist:
        val = g/(pixel_count/100)
        hist_norm.append(val)

    x = np.linspace(0,255,256)
    print(len(x))
    print(len(hist_norm))

    hist_img = plt.bar(x, hist_norm, 1, color='black')
    plt.show()

    asdf

    grid = np.asarray(img)
    grid.setflags(write=1)
    rx, ry = 12, 15 # block size

    bx, by = math.ceil(width/rx), math.ceil(height/ry)

    med_r = np.zeros(img.size, dtype="uint8")
    med_g = np.zeros(img.size, dtype="uint8")
    med_b = np.zeros(img.size, dtype="uint8")

    sat = np.zeros(img.size, dtype="uint8")

    hue_r = np.zeros(img.size, dtype="uint8")
    hue_g = np.zeros(img.size, dtype="uint8")
    hue_b = np.zeros(img.size, dtype="uint8")

    lum   = np.zeros(img.size, dtype="uint8")
    temp_hue = np.zeros(img.size, dtype="uint8")

    brightnesses = []

    for x in np.arange(0,width,width/rx):
        x = int(x)
        if x != 0:
            try:
                grid[:,x-2:x+2] = np.array([0,0,255])
            except IndexError:
                pass
        for y in np.arange(0,height,height/ry):
            y = int(y)
            if y != 0:
                try:
                    pass
                    grid[y-2:y+2,:] = np.array([0,0,255])
                except IndexError:
                    pass
            box = (x, y, x+bx, y+by)
            img_new = img.crop(box)
            r, g, b = ImageStat.Stat(img_new).median
            med_r[x:x+bx, y:y+by] = r
            med_g[x:x+bx, y:y+by] = g
            med_b[x:x+bx, y:y+by] = b
            lum[x:x+bx, y:y+by] = (r+g+b)/3
            brightnesses.append(sum(r+g+b)/3)
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            sat[x:x+bx, y:y+by] = s*255
            r, g, b = colorsys.hsv_to_rgb(h, 1, 255)
            hue_r[x:x+bx, y:y+by] = r
            hue_g[x:x+bx, y:y+by] = g
            hue_b[x:x+bx, y:y+by] = b
            temp_hue[x:x+bx, y:y+by] = 255


    num_bins = 8
    bin_width = 256/num_bins
    feature_vector = defaultdict(int)
    for i in range(0,8):
        feature_vector['brightness_bin_%i' %(i)] = 0
    
    for brightness in brightnesses:
        if brightness < bin_width:
            feature_vector['brightness_bin_%i' %(0)] += 1
        elif brightness > 256-bin_width:
            feature_vector['brightness_bin_%i' %(num_bins-1)] += 1
        else:
            location, remainder = divmod(brightness, bin_width)
            if remainder == 0:
                lower_bin = location
                feature_vector['brightness_bin_%i' %(lower_bin)] += 1
            else:
                lower_bin = math.floor(location)
                upper_bin = lower_bin + 1
                feature_vector['brightness_bin_%i' %(lower_bin)] += int(bin_width-remainder)
                feature_vector['brightness_bin_%i' %(upper_bin)] += int(remainder)


    med_r = med_r.T
    med_g = med_g.T
    med_b = med_b.T

    sat = sat.T

    hue_r = hue_r.T
    hue_g = hue_g.T
    hue_b = hue_b.T
    temp_hue = temp_hue.T

    lum = lum.T


    med_temp = np.dstack([med_r, med_g, med_b])
    sat_temp = np.dstack([sat, sat, sat])
    hue_temp = np.dstack([hue_r, hue_g, hue_b])
    lum_temp = np.dstack([lum, lum, lum])

    hue_r_arr = np.dstack([temp_hue, hue_g, hue_b])
    hue_g_arr = np.dstack([hue_r, temp_hue, hue_b])
    hue_b_arr = np.dstack([hue_r, hue_g, temp_hue])

    # temp = imresize(temp, img.size, interp='nearest')
    med_img = Image.fromarray(med_temp)
    sat_img = Image.fromarray(sat_temp)
    hue_img = Image.fromarray(hue_temp)
    lum_img = Image.fromarray(lum_temp)
    grid_img = Image.fromarray(grid)

    hue_img_r = Image.fromarray(hue_r_arr)
    hue_img_g = Image.fromarray(hue_g_arr)
    hue_img_b = Image.fromarray(hue_b_arr)



    plt.title('Main image')
    #plt.hist(grid.ravel(),256,(0,256))
    #plt.show()

    #plt.hist(lum_temp.ravel(),256,(0,255))
    #plt.show()
    #plt.hist(lum_temp.ravel(),8,(0,255))
    #plt.show()

    plt.title('Median image')
    #plt.hist(med_temp.ravel(),256,(0,256))
    #plt.show()

    plt.figure(1)
    plt.subplot(121)
    plt.title('Saturated image')
    plt.hist(sat_temp.ravel(),256,(0,256))
    plt.subplot(122)
    plt.title('Saturated image 8 Bins')
    plt.hist(sat_temp.ravel(),8,(0,256))
    plt.show()

    plt.figure(2)
    plt.subplot(121)
    plt.title('Brighntess image')
    plt.hist(lum_temp.ravel(),256,(0,255))
    plt.subplot(122)
    plt.title('Brighntess image 8 bins')
    plt.hist(lum_temp.ravel(),8,(0,255))
    plt.show()

    plt.figure(2)
    plt.subplot(421)
    plt.hist(hue_temp.flatten(),256,(0,256))
    #plt.show()
    plt.subplot(422)
    plt.hist(hue_temp.flatten(),8,(0,256))

    plt.subplot(423)
    plt.hist(hue_r_arr.flatten(),256,(0,256))
    #plt.show()
    plt.subplot(424)
    plt.hist(hue_r_arr.flatten(),8,(0,256))

    plt.subplot(425)
    plt.hist(hue_g_arr.flatten(),256,(0,256))
    #plt.show()
    plt.subplot(426)
    plt.hist(hue_g_arr.flatten(),8,(0,256))

    plt.subplot(427)
    plt.hist(hue_b_arr.flatten(),256,(0,256))
    #plt.show()
    plt.subplot(428)
    plt.hist(hue_b_arr.flatten(),8,(0,256))

    plt.show()


    med_img.save(os.path.basename(filename)+'_med'+'.jpg')
    sat_img.save(os.path.basename(filename)+'_sat'+'.jpg')
    hue_img.save(os.path.basename(filename)+'_hue'+'.jpg')
    lum_img.save(os.path.basename(filename)+'_lum'+'.jpg')

    hue_img_r.save(os.path.basename(filename)+'_hue_r'+'.jpg')
    hue_img_g.save(os.path.basename(filename)+'_hue_g'+'.jpg')
    hue_img_b.save(os.path.basename(filename)+'_hue_b'+'.jpg')

    grid_img.save(os.path.basename(filename)+'_grid'+'.jpg')

if __name__ == '__main__':
    args = docopt(__doc__)
    path = args['<PATH>']
    show = args['-s']
    extract_demo(path, show)
