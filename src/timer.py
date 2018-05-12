"""calculate time"""

import time
from datetime import datetime
from datetime import timedelta

class timer():
    def __init__(self):
        self.start_time = time.time()

    def get_duration(self):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        durations = timedelta(seconds=duration)
        print("It costs %s to run" % durations)
