class CustomDict:
    def __getitem__(self, key):
        key = str(hash(key))
        return getattr(self, key)

    def __setitem__(self, key, value):
        key = str(hash(key))
        setattr(self, key, value)


if __name__ == "__main__":
    d = CustomDict()
    d["x"] = 100
    d["y"] = 200
    d[1] = 300
    d[(1, 2)] = 400
    print(d["x"])
    print(d["y"])
    print(d[1])
    print(d[(1, 2)])
    print(d[4])
