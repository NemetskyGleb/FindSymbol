import random

filename = input("Enter name of file: ")
filesize = int(input("Enter size of file: "))

f = open(filename, "w")

charsStr = input("Enter characters to use through space:")

chars = charsStr.split(' ')

charsDict = {}

for x in range(filesize):
    char = random.choice(chars)
    if char in charsDict:
        charsDict[char] += 1
    else:
        charsDict[char] = 1 
    f.write(char)
