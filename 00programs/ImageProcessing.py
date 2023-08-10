import math

pixel0 = inputValue[0]
pixel1 = inputValue[1]
pixel2 = inputValue[2]

#darken
pixel0 = pixel0+3 if pixel0+3<20 else 20
pixel1 = pixel1+3 if pixel1+3<20 else 20
pixel2 = pixel2+3 if pixel2+3<20 else 20

#shift
pixelSpare = pixel0
pixel0 = pixel1
pixel1 = pixel0
pixel2 = pixelSpare

#average
sum = (pixel0+pixel1+pixel2)
average = math.floor(sum/3)
