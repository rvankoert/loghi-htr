import threading


class AppLocker:

    def __init__(self):
        self.value = 0
        self._lock = threading.Lock()
        self.processing = False

    def set_processing(self, value):
        # with self._lock:
        self.processing = value

    def get_processing(self):
        return self.processing
