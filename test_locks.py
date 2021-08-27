"""Mock-up to test object sharing in the chain lock mechanism."""

from time import perf_counter, sleep
from torch.multiprocessing import Process, set_start_method, Manager
from sdgai.utils import ChainLock


set_start_method('spawn', force=True)


class LockedClass:
    """Stand-in for a class, which should not have its data accessed concurrently."""

    N_WORKERS = 2

    def __init__(self, manager: Manager):
        self.lock = ChainLock(self.N_WORKERS, manager)
        self.data = manager.dict()


def worker_fn(num: int, locked_class: LockedClass):
    """Iteratively do something with the data of the given locked class."""

    sleep(num * 0.1)

    for j in range(5):
        with locked_class.lock:
            locked_class.data[perf_counter()] = (num, j, sum(locked_class.data.keys()))
            sleep(0.5 * (num + 1))

    sleep(num * 0.1)
    print(num)
    print(locked_class.data)


if __name__ == '__main__':
    manager = Manager()
    locked_class = LockedClass(manager)

    workers = [Process(target=worker_fn, args=(i, locked_class), daemon=True) for i in range(2)]

    for worker in workers:
        worker.start()
        sleep(0.05)

    for worker in workers:
        worker.join()

    print(locked_class.data)
    manager.shutdown()
