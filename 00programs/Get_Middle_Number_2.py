a = inputValue[0]

b = inputValue[1]

c = inputValue[2]

result = -1

result = a if (b<=a<=c) or (c<=a<=b) else result

result = a if (a <= b <= c) or (c <= b <= a) else result

result = c if (a<=c<=b) or (b<=c<=a) else result