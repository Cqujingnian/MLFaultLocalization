import math
grade = -9 #currently ungraded

average = math.floor(sum(inputValue)/8)

grade = 1 if average >=70 else grade
grade = 2 if (average >=60 and average<70) else grade
grade = 3 if (average >=50 and average<60) else grade
grade = 4 if (average >=40 and average<50) else grade
grade = 0 if average<40 else grade
grade = 0 if any(v<40 for v in inputValue) else grade
grade = 1 if (sorted(inputValue)[-1]>90 and sorted(inputValue)[-2]>90) else grade
grade = -1 if all(v==0 for v in inputValue) else grade

