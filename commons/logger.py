import sys
from datetime import datetime


class Logger(object):
    def __init__(self, logpath, syspart=sys.stdout):
        self.terminal = syspart
        self.log = open(logpath, "a")

    def write(self, message):

        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass

def log(*args):
    print(f'[{datetime.now()}]', *args)