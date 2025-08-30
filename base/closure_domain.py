x = 0  # 全局变量
z = 4  # 全局变量


def make_printer():
    global x  # 让 x 指向全局
    x = 5  # 修改了全局 x
    y = 8  # make_printer 的局部变量

    def printer():
        nonlocal x  # ❌ 报错，因为 x 不在上层局部作用域
        nonlocal y  # ✅ 绑定到 make_printer 的 y
        global z  # ✅ 使用全局变量 z

        x += 1
        y += 1
        z += 1
        print(x)

    return printer


if __name__ == "__main__":
    t = make_printer()
    t()
