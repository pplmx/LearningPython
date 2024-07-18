def debug(func):
    def wrapper(*args, **kwargs):
        print('Before the calling of the func')
        ret = func(*args, **kwargs)
        print('After the calling of the func')
        return ret

    return wrapper


def log(level):
    def out_wrapper(func):
        def inner_wrapper(*args, **kwargs):
            print(f'[{level}] Before the calling of the func')
            ret = func(*args, **kwargs)
            print(f'[{level}] After the calling of the func')
            return ret

        return inner_wrapper

    return out_wrapper


class DEBUG:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        print('Before the calling of the func')
        ret = self.func(*args, **kwargs)
        print('After the calling of the func')
        return ret


class LOG:
    def __init__(self, level):
        self.level = level

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            print(f'[{self.level}] Before the calling of the func')
            ret = func(*args, **kwargs)
            print(f'[{self.level}] After the calling of the func')
            return ret

        return wrapper


@debug
def hi():
    print('hi')


@log('INFO')
def hello():
    print('hello')


@DEBUG
def example():
    print('example')


@LOG('DEBUG')
def sample():
    print('sample')


def happy():
    print('happy')


if __name__ == '__main__':
    hi()
    hello()

    example()
    sample()

    # call it directly
    print('======= call it directly using func')
    debug(happy)()
    log('XXX')(happy)()

    # call with class
    print('======= call it directly using class')
    DEBUG(happy)()
    LOG('ZZZ')(happy)()
