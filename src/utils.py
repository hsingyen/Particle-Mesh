import time

class Timer:
    def __init__(self, label="Timer"):
        self.label = label
    def __enter__(self):
        self.start = time.time()
        return self
    def __exit__(self, *args):
        self.end = time.time()
        print(f"{self.label}: {self.end - self.start:.6f} seconds")