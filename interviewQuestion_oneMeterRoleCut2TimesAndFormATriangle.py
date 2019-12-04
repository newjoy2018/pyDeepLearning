import time
import random

testNum = 1000000

def test():
    m = random.random()
    n = random.random()
    a = min(m,n)
    b = max(m,n) - a
    c = 1-a-b
    if a+b <= c:
        return False
    elif a+c <= b:
        return False
    elif b+c <= a:
        return False
    else:
        return True

if __name__ == '__main__':
    start = time.time()
    success = 0
    for i in range(testNum):
        if test() == True:
            success += 1
    end = time.time()
    print('The probability of forming a triangle is', success/testNum)
    print('The calculation process takes', (end-start)*1000, 'ms')
