from __future__ import absolute_import, division, print_function
import os
from time import sleep
from random import random


class PathLockException(Exception):
    pass


class PathLock(object):

    lock_template = '%s.lock'

    def __init__(self, path, interval=1.0, repeat=-1):
        self.path = path
        self.lock_path = PathLock.lock_template % path
        self.locked = False
        self.interval = interval
        self.repeat = repeat

    def sleep(self):
        interval = random() * self.interval
        sleep(interval)

    def lock(self):
        assert not self.locked
        i = -1
        while self.repeat < 0 or i < self.repeat:
            i += 1
            try:
                with open(self.lock_path, 'x'):
                    pass
                self.locked = True
                return self
            except FileExistsError as ex:
                self.sleep()
                continue
        raise PathLockException()

    def unlock(self):
        assert self.locked
        assert os.path.exists(self.lock_path)
        os.remove(self.lock_path)
        self.locked = False

    def __enter__(self):
        return self.lock()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.locked:
            self.unlock()


def write(path, text, append=False, create_dirs=True):
    if create_dirs:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with PathLock(path):
        mode = 'a' if append else 'w'
        with open(path, mode) as f:
            f.write(text)


def append(path, text, create_dirs=True):
    write(path, text, append=True, create_dirs=create_dirs)
