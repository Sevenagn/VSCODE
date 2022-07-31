def conflict(state, nextX):
    nextY = len(state)
    for i in range(nextY):
        if abs(state[i] - nextX) in (0, nextY - i):
            return True
    return False


def queens(num=8, state=[]):
    for pos in range(num):
        if not conflict(state, pos):
            if len(state) < num - 1: 
                for result in queens(num, state + [pos]): #搜索
                    yield [pos] + result #回溯
            else:
                yield [pos] #边界
  

def prettyprint(no, solution):
    if no<2:
        return
    print('No.%d:'%no, solution)
    print('-' * 30)
    for pos in solution:
        print('. ' * (pos) + 'o ' + '. ' * (len(solution) - pos - 1))
    print('-' * 30)


result = list(queens(8))
no = 1
for i in result:
    prettyprint(no, i)
    no += 1
print('Total=%d'%len(result))