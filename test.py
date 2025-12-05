from math import cos, e
x_i = 0
x_d = 1

i = 0


def f(x):
    return cos(x) - x


fx_m = f((x_i+x_d)/2)

while (abs(fx_m) >= .001):
    x_m = (x_i+x_d)/2
    fx_m = f(x_m)
    print(
        f"Iteracion #{i}\n Xi = {x_i}, f(Xi) = {f(x_i)} \n Xd = {x_d}, f(Xd) = {f(x_d)} \n Xm = {x_m}, f(Xm) = {fx_m}")
    if fx_m >= 0:
        x_d = x_m
    else:
        x_i = x_m
    i += 1
