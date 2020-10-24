#Problem 1
#Introduction
#Hello

if __name__ == '__main__':
    print "Hello, World!"
  
#IfElse

  #!/bin/python3

import math
import os
import random
import re
import sys



if __name__ == '__main__':
    n = int(input().strip())
    if n%2 == 1 or 6 <= n <= 20:
        print("Weird")
    else:
        print("Not Weird")


#ArithmeticsOperations

if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a+b, a-b, a*b, sep = "\n")

#Division

if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a//b, a/b, sep = "\n")

#Loop

if __name__ == '__main__':
    n = int(input())
    for i in range(n):
        print(i*i, sep = "\n")

#WriteAFunction

def is_leap(year):
    leap = False
     
    if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
        leap = True
    else: 
        leap = False

    return leap

year = int(input())

#PrintFunction

if __name__ == '__main__':
    n = int(input())
    for i in range(1, n+1):
        print(i, end="")


#Data Types
#ListComprehension
if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())

    solutions = [[i, j, k] for i in range(x+1) for j in range(y+1) for k in range(z+1) if i+j+k != n]
    print(solutions)
    
#RunnerUpScore

if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())
    
    new_list=[]
    for i in arr:
        if i not in new_list:
            new_list.append(i)
    
    new_list.sort(reverse = True) 
    
    print(new_list[1])
    
#NestedList

if __name__ == '__main__':
 def secondLowestGrade(classList):
    secondLowestScore = sorted(set(_[1] for _ in classList))[1]
    result = sorted([_[0] for _ in classList if _[1] == secondLowestScore])
    return result


numberOfStudents = int(input())
classList = []
for i in range(numberOfStudents):
    classList.append([str(input()), float(input())])
print('\n'.join(secondLowestGrade(classList)))

#Lists

if __name__ == '__main__':
    N = int(input())
    newlist = []
    
    def result_handler(newlist):
        inp = input().split()
        action = inp[0]
        values = inp[1:]

        if action == "print":
            print(newlist)
        else:
            new_action = "newlist." + action + "("+ ",".join(values) +")"
            eval(new_action)

    for x in range(N):
        result_handler(newlist)
        
#FindPercentage

if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        grades = input().split()
        scores = list(map(float, grades[1:]))
        student_marks[grades[0]] = sum(scores) / 3
    #query_name = input()
    print("{:.2f}".format(student_marks[input()]))

#Tuples

if __name__ == '__main__':
    n = int(input())
    integer_list = map(int, input().split())
    t = tuple(integer_list)
    print(hash(t))
    
#String

#SwapCase

def swap_case(s):
    return s.swapcase()

if __name__ == '__main__':
    
#SplitAndJoin

def split_and_join(line):
    line = line.split(" ")
    line = "-".join(line)
    return line

if __name__ == '__main__':
    
#What's Your Name

def print_full_name(a, b):
    print("Hello " + a + " " + b + "! " + "You just delved into python.")

if __name__ == '__main__':

#Mutation

def mutate_string(string, position, character):
    l = list(string)
    l[position] = character
    string = "".join(l)
    return string

if __name__ == '__main__':
    
#FindAString

def count_substring(string, sub_string):
    res = 0
    sub_len = len(sub_string)
    for i in range(len(string)):
        if string[i:i+sub_len] == sub_string:
            res += 1
    return res

if __name__ == '__main__':
    
#String Validators

if __name__ == '__main__':
    s = input()
    print(any([char.isalnum() for char in s]))
    print(any([char.isalpha() for char in s]))
    print(any([char.isdigit() for char in s]))
    print(any([char.islower() for char in s]))
    print(any([char.isupper() for char in s]))
    
#Text Alignment

#Replace all ______ with rjust, ljust or center. 

thickness = int(input()) #This must be an odd number
c = 'H'

#Top Cone
for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))

#Top Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))

#Middle Belt
for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))    

#Bottom Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))    

#Bottom Cone
for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))

#Text Wrap

import textwrap

def wrap(string, max_width):
    return textwrap.fill(string, max_width)

if __name__ == '__main__': 
    
#Design Door Mat

height, length = map(int, input().split())

for i in range(0, height // 2):
    s = '.|.' * (i * 2 + 1)
    print(s.center(length,'-'))
print('WELCOME'.center(length, '-'))
for i in range(height // 2 - 1, -1, -1):
    s = '.|.' * (i * 2 + 1)
    print(s.center(length,'-'))  
    
#String Formatting

def print_formatted(number):
    width = len("{0:b}".format(number))
    for i in range(1, number + 1):
        print("{0:{width}d} {0:{width}o} {0:{width}X} {0:{width}b}".format(i, width = width))

if __name__ == '__main__':
    
#Capitalize!
def solve(s):
    string = " ".join(i.capitalize() for i in s.split(" "))
    return string

#The Minion's Game

def minion_game(string):
    s_len = len(string)
    kevin = 0
    stuart = 0
    vowels = "AEIOU"

    for x in range(s_len):
        if string[x] in vowels:
            kevin += s_len - x
        else: 
            stuart += s_len - x
    #sono preferibili condizioni a cascata tramite elif
    if kevin > stuart:
        print("Kevin", kevin)
    elif kevin < stuart:
        print("Stuart", stuart)
    else:
        print("Draw")

    


if __name__ == '__main__':
    s = input()
    minion_game(s) 
    
#sets
#ntroduction to sets
def average(array):
    newset = set(array)
    average = sum(newset) / len(newset)
    return average
if __name__ == '__main__':
    
#No Idea
n = input()
array = input().split()
A = set(input().split())
B = set(input().split())
happiness = 0
sadness = 0

for i in array:
    if i in A:
        happiness += 1
    elif i in B:
        sadness += 1
print(happiness - sadness)

#set Add
print(len(set(input() for i in range(int(input())))))

#Remove Discard Pop

n = int(input())
s = set(map(int, input().split()))

for i in range(int(input())): 
    command = input().split() #comandi inseriti
    if command[0] == "pop":
        s.pop()
    elif command[0] == "remove":
        s.remove(int(command[1]))
    elif command[0] == "discard":
        s.discard(int(command[1]))
    
print(sum(s))

#Union
ne = int(input())
eng = set(map(int, input().split()))
nf = int(input())
fre = set(map(int, input().split()))

print(len(eng.union(fre)))

#Inteersection

ne = int(input())
eng = set(map(int, input().split()))
nf = int(input())
fre = set(map(int, input().split()))

print(len(eng.intersection(fre)))

#Difference

ne = int(input())
eng = set(map(int, input().split()))
nf = int(input())
fre = set(map(int, input().split()))

print(len(eng.difference(fre)))

#symmetric Difference

ne = int(input())
eng = set(map(int, input().split()))
nf = int(input())
fre = set(map(int, input().split()))

print(len(eng.symmetric_difference(fre)))

#Mutation

def mutateSet(A):
    command = input().split()[0]
    other_set = set(map(int, input().split()))
    if command == "update":
        A.update(other_set)
    if command == "intersection_update":
        A.intersection_update(other_set)
    if command == "difference_update":
        A.difference_update(other_set)
    if command == "symmetric_difference_update":
        A.symmetric_difference_update(other_set)

n = input() 
A = set(map(int, input().split()))
for i in range(int(input())):
    mutateSet(A)
print(sum(A))

#Captain's room

g_size = int(input())
room_list = list(map(int, input().split()))

captain = (sum(set(room_list)) * g_size - sum(room_list)) // (g_size-1)
print(captain)

#Check Subset
test_cases = int(input())
for i in range(test_cases):
    na = int(input())
    a = set(map(int, input().split()))
    nb = int(input())
    b = set(map(int, input().split()))
    print(a.issubset(b))
    
#Superset

a = set(map(int, input().split()))
n_sub= int(input())
n = set(map(int, input().split()))
print(all(a>n for i in range(n_sub)))

#symmetric difference

M = int(input())
m = set(map(int, input().split()))
N = int(input())
n = set(map(int, input().split()))

print(*sorted(m.symmetric_difference(n)), sep ="\n")

#Collections 
#>collections Counter

import collections;
from collections import Counter

n_shoes = int(input())
stock = collections.Counter(map(int, input().split()))
customers = int(input())
total = 0

for n in range(customers):
   size, price = map(int, input().split())
   if stock[size]:
    total = total + price
    stock[size] -= 1
print(total)

#Default dict

from collections import defaultdict
n, m = map(int, input().split())
d = defaultdict(list)

for i in range(1, n+1):
    d[input()].append(str(i))
for j in range(m):
    print(' '.join(d[input()]) or -1)
    
#Collection namedtuple
from collections import namedtuple
n_students, Columns = int(input()), namedtuple("n_students", input())

print("{:.2f}".format(sum([int(Columns(*input().split()).MARKS) for x in range(n_students)]) / n_students))

#Word order

from collections import Counter, OrderedDict
class OrderedCounter(Counter, OrderedDict):
    pass

ordered_counter = OrderedCounter(input() for i in range(int(input())))

print(len(ordered_counter))
print(*ordered_counter.values())

#Collection deque

from collections import deque
d = deque()

for i in range(int(input())):
    command = input().split()
    if command[0] == "append":
        d.append(command[1])
    if command[0] == "pop":
        d.pop()
    if command[0] == "popleft":
        d.popleft()
    if command[0] == "appendleft":
        d.appendleft(command[1])
print(*d)

#Company logo 

from collections import Counter, OrderedDict
class orderedCounter (Counter, OrderedDict):
    pass

ordered_counter = orderedCounter(sorted(input())).most_common(3)
[print(*word) for word in ordered_counter]

#Pilling up

from collections import deque

for i in range(int(input())):
    cubesn, cubesl = int(input()), deque(map(int, input().split()))
    pile = True
    for j in range(len(cubesl)-1):
        if cubesl[0] >= cubesl[1]:
            cubesl.popleft()
        elif cubesl[-1] >= cubesl[-2]:
            cubesl.pop()
        else:
            pile = False

    if pile:
        print('Yes')
    else:
        print('No') 
#Calendar module
import calendar
month, day, year = map(int, input().split())
current_day = calendar.weekday(year, month, day)
print(calendar.day_name[current_day].upper())

#Time delta

import math
import os
import random
import re
import sys
from datetime import datetime as dt

# Complete the time_delta function below.
fmt = '%a %d %b %Y %H:%M:%S %z'
for t_itr in range(int(input())):
    t1 = dt.strptime(input(), fmt)
    t2 = dt.strptime(input(), fmt)
    print(int(abs(t1-t2).total_seconds()))


#Exception
for i in range(int(input())):
    a, b = input().split()
    try:
        print(int(a) // int(b))
    except ZeroDivisionError as e:
        print('Error Code:', e)
    except ValueError as v:
        print('Error Code:', v)

#Built in
n, x = input().split()
marks = []
for i in range(int(x)):
    marks.append(map(float, input().split()))
for i in zip(*marks):
    print(sum(i)/len(i))  

#Athlete Sort
n, m = list(map(int, input().split()))
athlete = [list(map(int, input().split())) for x in range(n)]
k = int(input())

sorted_athlete = sorted(athlete, key = lambda i: i[k])
for j in sorted_athlete:
    print(*j)
#ginortS
line = input()
a, b, c, d = [], [], [], []

for i in sorted(line):
    if i.islower():
        a.append(i)   
    if i.isupper():
        b.append(i)
    if i.isdigit():
        if int(i) % 2 !=0:
            c.append(i)
        else:
            d.append(i)

print(''.join(sorted(a)) + ''.join(sorted(b)) + ''.join(c) + ''.join(d))

#map and lambda functions 
cube = lambda x: x**3

def fibonacci(n):
    # return a list of fibonacci numbers
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-2] + fib[i-1])  
    return(fib[0:n])

#detect floating point number
t = int(input())
def isfloat(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

s = [input().strip() for i in range(t)]
for j in s:
    if j == '0':
        print(False)
    else:
        print(isfloat(j))
        
#regex
#re.split()
regex_pattern = (r"[,.]")	

import re
print("\n".join(re.split(regex_pattern, input())))

#group() groups() groupdict()
import re
m = re.search(r'([a-zA-Z0-9])\1+', input())
if m:
    print(m.group(1))
else:
    print('-1')

#re.findall()
import re
c = '[qwrtypsdfghjklzxcvbnm]'
m = re.findall('(?<=' + c +')([aeiou]{2,})' + c, input(), re.I)
print('\n'.join(m or ['-1']))

#re.start() re.end()
import re
s = input()
k = input()
if k in s:
    print(*[(i.start(), (i.start()+len(k)-1)) for i in re.finditer(r'(?={})'.format(k), s)], sep='\n')
else:
    print('(-1, -1)')
#regex substitution
import re
[print(re.sub('(?<=\s)\&\&\s', 'and ', re.sub('\s\|\|\s', ' or ', input()))) for _ in range(int(input()))]

#validating roman numbers 
thousands = '(?:(M){0,3})?'
hundreds = '(?:(D?(C){0,3})|(CM)|(CD))?'
tens = '(?:(L?(X){0,3})|(XC)|(XL))?'
numbers = '(?:(V?(I){0,3})|(IX)|(IV))?'
regex_pattern = r'^' + thousands + hundreds + tens + numbers + '$'	# Do not delete 'r'
import re
print(str(bool(re.match(regex_pattern, input()))))

#validating phone numbers 
import re
n = int(input())
for i in range(n):
    if re.match(r'[789]\d{9}$', input()):
        print('YES')
    else:
        print('NO')
        
#validating and parsing email adresses
import re
n = int(input())

for i in range(n):
    name, email = input().split()
    result = re.match(r'<[A-Za-z](\w|-|\.|_)+@([A-Za-z]+\.[A-Za-z]{1,3})>', email)
    if result:
        print(name, email)
#hex color code
import re
n = int(input())
for i in range(n):
    line = input()
    m = re.findall(r'(?<!^)(#(?:[\da-f]{3}){1,2})', line, re.I)
    for i in m:
        print(i)
        
#html parser part1
from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print('Start :', tag)
        for i in attrs:
            print('->', i[0], '>', i[1])
    def handle_endtag(self, tag):
        print('End   :', tag) 
    def handle_startendtag(self, tag, attrs):
        print('Empty :', tag)
        for j in attrs:
            print('->', j[0], '>', j[1]) 

parser = MyHTMLParser()
for i in range(int(input())):
    parser.feed(input())
    
#html parser part 2 
from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_comment(self, data):
        if '\n' in data:
            print('>>> Multi-line Comment')
        else:
            print('>>> Single-line Comment')
        print(data)    
    def handle_data(self, data):
        if data == '\n':
            return
        print('>>> Data')
        print(data) 
  
html = ""       
for i in range(int(input())):
    html += input().rstrip()
    html += '\n'
    
parser = MyHTMLParser()
parser.feed(html)
parser.close()

#Detect HTML Tags, Attributes and Attribute Values

from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print(tag)
        for i, value in attrs:
            print("->", i, ">", value)

    def handle_startendtag(self, tag, attrs):
        print(tag)
        for i, value in attrs:
            print("->", i, ">", value)

html = ''
for _ in range(int(input())):
    html += input().rstrip()
    html += '\n'

parser = MyHTMLParser()
parser.feed(html)
parser.close()

#uid validation
import re
for i in range(int(input())):
    lines = input()
    if re.match(r'^(?!.*(.).*\1)(?=(?:.*[A-Z]){2,})(?=(?:.*\d){3,})[a-zA-Z0-9]{10}$', lines):
        print('Valid')
    else:
        print('Invalid')
#validating credit card number
import re
for i in range(int(input())):
    lines = input().rstrip()
    if re.search(r'^[456]\d{3}(\?|(-)|)(\d{4}\1){2}\d{4}$', lines) and not re.search(r'(\d)(-|\1){4}', lines):
        print('Valid')
    else:
        print('Invalid')
#validating postcode
regex_integer_in_range = r"^[1-9][0-9]{5}$"	# Do not delete 'r'.
regex_alternating_repetitive_digit_pair = r"(.)(?=.\1)"	# Do not delete 'r'.
import re
P = input()
print (bool(re.match(regex_integer_in_range, P)) 
and len(re.findall(regex_alternating_repetitive_digit_pair, P)) < 2)

#matrix script
#!/bin/python3

import math
import os
import random
import re
import sys


n,m = list(map(int, input().split()))
matrix = [input() for i in range(n)]
result = ''.join([j[i] for i in range(m) for j in matrix])
result = re.sub('([A-Za-z1-9])[^A-Za-z1-9]+([A-Za-z1-9])', r'\1 \2', result)
result = re.sub('  ', ' ', result)
print(result)

#XML
#xml1 find the score 
import sys
import xml.etree.ElementTree as etree

def get_attr_number(node):
   return sum(len(i.attrib) for i in node.iter())


if __name__ == '__main__':
    sys.stdin.readline()
    xml = sys.stdin.read()
    tree = etree.ElementTree(etree.fromstring(xml))
    root = tree.getroot()
    print(get_attr_number(root))
    
#xml2 find the maximun depth 
import xml.etree.ElementTree as etree

maxdepth = 0
def depth(elem, level):
    global maxdepth
    if(level == maxdepth):
        maxdepth += 1
    for i in elem:
        depth(i, level +1)    
    return maxdepth
if __name__ == '__main__':
    n = int(input())
    xml = ""
    for i in range(n):
        xml =  xml + input() + "\n"
    tree = etree.ElementTree(etree.fromstring(xml))
    depth(tree.getroot(), -1)
    print(maxdepth)
    
#decorators 
#standardize mobile numbers using decorators
def wrapper(f):
    def fun(l):
        f(['+91 '+ i[-10:-5]+ ' ' + i[-5:] for i in l])          
    return fun

@wrapper
def sort_phone(l):
    print(*sorted(l), sep='\n')

if __name__ == '__main__':
    l = [input() for _ in range(int(input()))]
    sort_phone(l) 
    
#numpy
#arrays
import numpy
def arrays(arr):
    return  numpy.array(arr[::-1], float)
arr = input().strip().split(' ')
result = arrays(arr)
print(result)

#shape and reshape
import numpy
n = numpy.array(input().split(), int)
print(numpy.reshape(n, (3,3)))

#transpose and flatten
import numpy
n, m = input().split()
arrays = numpy.array([input().split() for m in range(int(n))], int)
print(numpy.transpose(arrays))
print(arrays.flatten())

#concatenate
import numpy
n, m, p = map(int, input().split())
array1 = numpy.array([input().split() for p in range(n)], int)
array2 = numpy.array([input().split() for p in range(m)], int)
print(numpy.concatenate((array1, array2), axis=0))

#zeroes and ones
import numpy as np 
values = list(map(int, input().split()))
print(np.zeros((values), dtype = np.int))
print(np.ones((values), dtype = np.int))

#eye and identity
import numpy
n, m = map(int, input().split())
numpy.set_printoptions(legacy='1.13') #change print options 
print(numpy.eye(n, m, k=0))

#array mathematics
import numpy 
n, m = map(int, input().split())
a = numpy.array([input().split() for m in range(n)], int)
b = numpy.array([input().split() for m in range(n)], int)
print(numpy.add(a,b))
print(numpy.subtract(a,b))
print(numpy.multiply(a,b))
print(a//b)
print(numpy.mod(a,b))
print(numpy.power(a,b))

#floor ceil and rint
import numpy as np
array = list(input().split())
np_array = np.array(array, float)
np.set_printoptions(legacy='1.13') #change print options 
print(np.floor(np_array))
print(np.ceil(np_array))
print(np.rint(np_array))

#sum and prod
import numpy as np
n, m = map(int, input().split())
arrays = np.array([input().split() for m in range(n)], int)
summed_result = np.sum(arrays, axis=0)
print(np.prod(summed_result))

#min and max
import numpy as np
n, m = map(int, input().split())
arrays = np.array([input().split() for m in range(n)], int)
minimun = np.min(arrays, axis=1)
print(np.max(minimun))

#mean, var and std
import numpy as np
n, m = map(int, input().split())
np.set_printoptions(legacy='1.13') #change print options 
arrays = np.array([input().split() for m in range(n)], int)
print((np.mean(arrays, axis=1)),(np.var(arrays,axis=0)), (np.std(arrays)), sep='\n')

#dot and cross
import numpy as np
n = int(input())
A = np.array([input().split() for i in range(n)], int)
B = np.array([input().split() for i in range(n)], int)
print(np.dot(A,B))

#inner and outer
import numpy as np
A = np.array(input().split(), int)
B = np.array([input().split()], int)
print(*np.inner(A,B), np.outer(A,B), sep='\n')

#polynomials
import numpy as np
p = np.array(input().split(), float)
print(np.polyval(p, float(input())))

#linear algebra
import numpy as np
n = int(input())
A = np.array([input().split() for i in range(n)], float)
print(round(np.linalg.det(A), 2))


#problem 2


#birthday cake candles
#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'birthdayCakeCandles' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER_ARRAY candles as parameter.
#

def birthdayCakeCandles(candles):
    count = candles.count(max(candles))
    return count
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    candles_count = int(input().strip())

    candles = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(candles)

    fptr.write(str(result) + '\n')

    fptr.close()
    
#Number Line Jumps

import math
import os
import random
import re
import sys

# Complete the kangaroo function below.
def kangaroo(x1, v1, x2, v2):
    if v1 > v2:
        if (x1-x2) % (v2-v1) == 0:
            return 'YES'
    
    return 'NO'


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    x1V1X2V2 = input().split()

    x1 = int(x1V1X2V2[0])

    v1 = int(x1V1X2V2[1])

    x2 = int(x1V1X2V2[2])

    v2 = int(x1V1X2V2[3])

    result = kangaroo(x1, v1, x2, v2)

    fptr.write(result + '\n')

    fptr.close()
    
#Viral advertising
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the viralAdvertising function below.
def viralAdvertising(n):
    count = [2]
    for i in range(n-1):
        count.append(count[i]*3//2)
    return sum(count)
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input())

    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')

    fptr.close()
    
#<recursive digit sum
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the superDigit function below.
def superDigit(n, k):
   l = map(int, list(n))
   return getSuperDigit(str(sum(l) * k))
    
def getSuperDigit(x):
    if len(x) == 1:
        return int(x)
    else:
        l = map(int, list(x)) 
        return getSuperDigit(str(sum(l)))

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    nk = input().split()

    n = nk[0]

    k = nk[1]

    result = superDigit(n, k)

    fptr.write(str(result) + '\n')

    fptr.close()
#insertion sort part1

import math
import os
import random
import re
import sys

# Complete the insertionSort1 function below.
def insertionSort(arr):
    j = arr[-1]
    for i in range(n-2, -1, -1):
        if arr[i] > j:
            arr[i+1] = arr[i]
            print(" ".join(str(x) for x in arr))
        else:
            arr[i+1] = j
            print(" ".join(str(x) for x in arr))
            return
    arr[0] = j
    print(" ".join(str(x) for x in arr))
    return

n = int(input())
arr = [int(i) for i in input().strip().split()]
insertionSort(arr)

#insertion sort part2

import math
import os
import random
import re
import sys

# Complete the insertionSort2 function below.
def insertionSort2(n, arr):
    for i in range(1, n):
        x = arr[i]
        j = i-1
        while j >= 0 and arr[j] > x:
            arr[j+1] = arr[j]
            j = j-1
        arr[j+1] = x
        print(' '.join(str(y) for y in arr))


if __name__ == '__main__':
    n = int(input())

    arr = list(map(int, input().rstrip().split()))

    insertionSort2(n, arr)