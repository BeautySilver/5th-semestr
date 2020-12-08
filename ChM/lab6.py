# задача Коші, модифицированный метод Ейлера

# y' = -x - 2xy
# y(1) = 1, h = 1

yy = lambda x,y : -x - 2*x*y
xn = 1
yn = 1

for i in range(10):
    y_h = yn + yy(xn, yn)
    y_n1 = yn + (yy(xn, yn) + yy(xn+1, y_h))/2

    print(y_n1)

    xn += 1
    yn = y_n1