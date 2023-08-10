import random
import math

unit = random.randrange(1,3+1)
value = 0
conversion = 0
if unit==1:
    value = random.randrange(-5,30)
    conversion = math.floor((value/5)*9+32)
if unit==2:
    value = random.randrange(27,50)
    conversion = math.floor(((value-32)*5)/9)
if unit==3:
    value = random.randrange(40,60)
    conversion = math.floor(value/39)

inputTC = [unit,value]
outputTC = conversion
