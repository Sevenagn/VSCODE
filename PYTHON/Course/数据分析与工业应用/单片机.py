import serial

s = serial.Serial('COM1',9600,timeout=1)

s.write(b'startA')

f = open('a.csv','wb') #写二进制

while True:
    line = s.readline()

    print(line) 
    if line == b'end':
        break

    f.write(line)
s.close()