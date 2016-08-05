'''
Sample code using numpy to convert image to array
Author: Win Woo
'''

import numpy
from PIL import Image
from numpy import array

def main():
    img = Image.open('/tmp/normal-0.jpg')
    width, height = img.size

    # crop out the center 300x300
    crop_h = (width-150)/2
    crop_v = (height-150)/2
    img = img.crop((crop_h, crop_v, width-crop_h, height-crop_v))

    # resize the resulting image to 150x150
    img = img.resize((150, 150))

    # convert to grayscale
    img = img.convert('L')
    img.save("/tmp/new.jpg")
    arr = array(img).reshape(150*150)
    print(arr)

if __name__ == '__main__':
    main()
