import time


def seconds():
    return round(time.time(), 30)


class TimeQuantizer:
    def __init__(self):
        self.start_time = seconds()
        self.intermediate_time = seconds()
        self.tick_time = seconds()

    def measure_step(self):
        previous = self.intermediate_time
        self.intermediate_time = seconds()
        return self.intermediate_time - previous

    def measure_total(self):
        return seconds() - self.start_time

    def set_tick(self):
        self.tick_time = seconds()

    def measure_tick(self):
        return seconds() - self.tick_time

    def reset(self):
        self.start_time = seconds()
        self.intermediate_time = seconds()
