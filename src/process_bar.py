import sys
import math

class process_bar():
    def __init__(self, max_step):
        self.i = 0
        self.max_step = int(max_step)
        self.max_arrow = 50

    def show_process(self, i=None):
        self.i = self.i+1

        now_p = self.i if self.i <= self.max_step else self.max_step
        all_p = self.max_step
        max_arrow = self.max_arrow
        percent = now_p * 100.0 / all_p
        assert now_p <= all_p

        num_arrow = math.ceil(now_p/all_p*max_arrow)
        process_bar_ = '[' + '>' * num_arrow + '-' * (max_arrow-num_arrow) + ']' \
                + '%.2f' % percent + '%' + '\r'
        if self.i == self.max_step:
            process_bar_ += '\n'
        sys.stdout.write(process_bar_)
        sys.stdout.flush()

    def reset(self):
        self.i = 0

    def set_maxstep(self, max_step_):
        self.max_step = max_step_

