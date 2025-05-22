from Main import FARIS

faris = FARIS()


cls, reasoning, audio = faris.predict([  5,   6,   0,   1,   3,  18,   1,   0,   8,   1,  44,  38,   2,  196,   0,   3,  12, 161,   9,   0,   0,   0])

print(cls, reasoning, audio)
print('perfecto')