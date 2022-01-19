import timeit


class Timer:
    def __init__(self):
        self.clock = {}

    def start(self, key="default"):
        self.clock[key] = timeit.default_timer()

    def end(self, key="default"):
        if key not in self.clock:
            raise Exception(f"{key} is not in the clock.")
        interval = timeit.default_timer() - self.clock[key]
        del self.clock[key]
        return interval
