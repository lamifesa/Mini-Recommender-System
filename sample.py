dic = {0:'a', 1:['b',4], 2:'c', 4:'d'}
# my = {}
# my.update
# print(my)

count=0
for key, value in dic.items():
    if count == 2:
        break
    else:
        print(value)
        count += 1