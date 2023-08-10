import random
def get_middle_number(a):
    # 判断 a 是否为中间数字
    result = -1
    if (a[1] <= a[0] <= a[2]) or (a[2] <= a[0] <= a[1]):
        result = a[0]
    # 判断 b 是否为中间数字
    if (a[0] <= a[1] <= a[2]) or (a[2] <= a[1] <= a[0]):
        result = a[1]
    # c 为中间数字
    if(a[0] <= a[2] <= a[1]) or (a[1] <= a[2] <= a[0]):
        result = a[2]
    return result

inputTC = [random.randint(0,10) for i in range(3)]
outputTC = get_middle_number(inputTC)
# print(outputTC)
