def f(a, b, c, d):
    ### forward pass ###
    e = a * b
    f = c + d
    g = e * f
    h = exp(g)
    L = 1 / h

    ### backward pass ###
    grad_L = 1
    grad_h = -1 / (h ** 2) * grad_L

    grad_g = exp(g) * grad_h
    grad_e = f * grad_g
    grad_f = e * grad_g
    grad_a = b * grad_e
    grad_b = a * grad_e
    grad_c = 1 * grad_f
    grad_d = 1 * grad_f

    return L, (grad_a, grad_b, grad_c, grad_d)


def f(x, y, z):
    ### forward pass ###
    a = 2 * x
    b = 1 / x
    c = 3 * y * z
    d = a - b
    e = d + c
    f = exp(e)
    g = f + 1
    L = 1 / g

    ### backward pass ###
    grad_L = 1
    grad_g = -1 / (g ** 2) * grad_L

    grad_f = 1 * grad_g
    grad_e = exp(e) * grad_f

    grad_d = 1 * grad_e
    grad_c = 1 * grad_e

    grad_a = 1 * grad_d
    grad_b = -1 * grad_d

    grad_x1 = 2 * grad_a
    grad_x2 = -1 / (x ** 2) * grad_b
    grad_x = grad_x1 + grad_x2

    grad_y = 3 * z * grad_c
    grad_z = 3 * y * grad_c

    return L, (grad_x, grad_y, grad_z)