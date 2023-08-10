import random
import math

input = []
input.append(random.randrange(0,20))
input.append(random.randrange(0,20))
input.append(random.randrange(0,20))
inputTC = input

pixel0 = input[0]
pixel1 = input[1]
pixel2 = input[2]

#darken
pixel0 = pixel0+3 if pixel0+3<20 else 20
pixel1 = pixel1+3 if pixel1+3<20 else 20
pixel2 = pixel2+3 if pixel2+3<20 else 20

#shift
pixelSpare = pixel0
pixel0 = pixel1
pixel1 = pixel2
pixel2 = pixelSpare

#average
sum = (pixel0+pixel1+pixel2)
outputTC = math.floor(sum/3)
