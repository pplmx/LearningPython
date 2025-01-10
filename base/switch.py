#!/usr/bin/env python


switch = {
    "add": lambda x, y: x + y,
    "sub": lambda x, y: x - y,
    "mul": lambda x, y: x * y,
    "div": lambda x, y: x / y,
}

if __name__ == "__main__":
    print(switch["add"](1, 8))
    print(switch["sub"](1, 8))
    print(switch["mul"](1, 8))
    print(switch["div"](1, 8))
