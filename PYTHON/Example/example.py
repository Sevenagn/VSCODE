#print('hello world!')
# a=400

# def test2():
#     a=300
#     print('----test2----修改前---a=%d,id=%d'%(a,id(a)))
#     if 1>2:
#         b=100
#         print(a)
#     else:
#         b='a'
#     print(b)


# def test1():
#     global a
#     # a=100
#     print('----test1----修改前---a=%d,id=%d'%(a,id(a)))
#     a=200
#     # print('----test1----修改前---a=%d,id=%d'%(a,id(a)))
#     # test2()
# #print(test1)
# a=400
# print('----test1----修改前---a=%d,id=%d'%(a,id(a)))




# test1()
#递归阶乘
# def calnum(n):
#     # if not n.isdigit():
#     #     return '请输入数字'
#     if n <= 1:
#         return 1
#     return n * calnum(n - 1)

# print(calnum('n'))

#斐波那契数列
# def fib(n):
#     if n==1:
#         return 1
#     if n==2:
#         return 1
#     return fib(n-1)+fib(n-2)
# print (fib(5))


#汉诺塔
# def hanoi(n, a, b, c):
#     global step
    
#     if n == 1:
#         step +=1
#         print(str(step),a, '-->', c)
        
#     else:
#         hanoi(n - 1, a, c, b)
#         step +=1
#         print(str(step),a, '-->', c)
#         hanoi(n - 1, b, a, c)
# # 调用
# step=0
# hanoi(3, 'A', 'B', 'C')


# f=open('test.txt','w')
# f.write('hello world!')
# f.close


# f=open('test.txt','r')
# print(f.read())
# f.close


# import os
# print(os.listdir)

# class Car:
#     def __init__(self,x,y):
#         self.x=x
#         self.y=y
#     def __new__(cls: type[Self]) -> Self:
#         pass
#     def move():
#         print('--------------')
# car=Car(1,2)
# print(id(car))

#斐波那契数列
# def fib(n):   #递归实现           
#     if n==1:
#         return 5
#     if n==2:
#         return 1
#     return fib(n-1)+fib(n-2)
# print (fib(5))

# def fib(n):   #循环实现
#     a,b=0,1
#     for i in range(n):
#         a,b=b,a+b
#     return a
# print (fib(5))

#汉诺塔
def hanoi(n, a, b, c):
    if n == 1:
        print(a, '-->', c)
    else:
        hanoi(n - 1, a, c, b)
        print(a, '-->', c)
        hanoi(n - 1, b, a, c)
hanoi(3, 'A', 'B', 'C')



