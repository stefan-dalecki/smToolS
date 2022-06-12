class Bright:
    def __init__(self, method):
        self._options = [
            option for option in dir(Bright) if option.startswith("__") is False
        ]
        assert method in self._options, "Chosen method is not available"
        self(method)

    def __call__(self, method):
        func = getattr(Bright, method)
        func(self, "test")

    def automatic(self, name):
        print(name)
        print("auto")

    def manual(self):
        print("man")

    def none(self):
        print("none")


b = Bright("automatic")
