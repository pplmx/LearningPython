x = 0
z = 4


def make_printer():
    global x
    x = 5
    y = 8

    def printer():
        nonlocal x
        nonlocal y
        global z
        x += 1
        y += 1
        z += 1
        print(x)

    return printer


if __name__ == "__main__":
    t = make_printer()
    t()
