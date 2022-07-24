# 变量和简单数据类型
## 字符串
### print() 输出
### str() 转换为字符串
### title() 首字母大写
### upper() 转换为大写
### lower() 转换为小写
### strip() 删除左右空格
### rstrip() 删除右空格
### lstrip() 删除左空格
## 转义字符
### \t 制表符
### \n 回车符
## 数字
### ** 使用两个乘号表示乘方运算
## 注释
### 字符#  
# 列表
## 访问列表元素
### list[0] 访问第一个列表元素
### list[-1] 访问最后一个列表元素
## 增删改列表元素
### list.append('add para') 在列表末增加元素
### list.insert(0,'insert para') 在列表头插入 元素
## 删除列表元素
### del list[0] 删除列表第一个元素
### 
## 修改列表元素
### list[0]='modify' 将第一个元素修改为modify
## 排序
### cars.sort() 永久性排序(按字母)
### cars.sort(reverse=True) 永久性排序(按字母相反的顺序)
### print(shorted(cars)) 临时排序，不改变原列表的顺序
### cars.reverse() 倒着打印列表(反转顺序，不一定是按字母排序)
### len(cars) 获取列表的长度
## 循环
### for magician（每次循环取得的值变量） in magicians:(列表)
### 缩进 print(magician) //循环执行
### print（"结束"） //只执行一次
## 创建数字列表
### for  value in range(1,5):
### 缩进 print(value)
### 输出 1~4
### numbers=list(range(1,7,2)) //从1到6生成列表，每次加2
### print(numbers)
### 对列表计算
### min(列表) 取最小
### max(列表) 取最大
### sum(列表) 求和
## 切片
### players = ['1','2','3','4','5','6']
### print(players[0:4]) //输出1~3的元素
### 使用切片复制能得到两个列表，可以各自改变这两个列表的值，而用赋值则这两个列表是一样的
## 元组
### 不可变的列表称为元组
### dimensions=(7,77) 元组是()，列表是[]
### 元组的元素不能修改，但是可以给元组的变量重新赋值
### dimensions[0]=1 //不行
### dimensions=(1,11) //可以
# if语句
### if car='bmw':
### 缩进 print('bmw')
### else:
### 缩进 print('others')
### 等于 == (区分大小写)
### 不等于 !=