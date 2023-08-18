import time


class Timer:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.time_sta = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        elapsed_time = time.perf_counter() - self.time_sta
        print(f"{self.label}: {elapsed_time}")