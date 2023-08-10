import math
# 1: CtoF; 2: FtoC; 3: inches to m; 

unit = inputValue[0]
value = inputValue[1]

answer = 0
answer = math.floor(((value/5)*9+32)) if unit==1 else answer
answer = math.floor((((value-32)*9)/5)) if unit==2 else answer
answer = math.floor(value/39) if unit==3 else answer
