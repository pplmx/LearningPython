#!/usr/bin/env python
import os
import random
import time
from multiprocessing import Pool, Process, Queue


# create a multiprocessing pool
def create_pool(num_processes: int) -> Pool:
    """
    Create a multiprocessing pool.
    """
    pool = Pool(processes=num_processes)
    return pool


def long_time_task(name: str) -> None:
    """
    A function to be run in a process, who needs much time to finish.
    """
    print(f"Run task {name}[{os.getpid()}]...")
    start = time.time()
    time.sleep(random.random() * 3)
    cost = time.time() - start
    print(f"Task {name} runs {cost:.2f} seconds.")


def test_multiprocess():
    """
    Test multiprocessing.
    """
    print("Parent process %s." % os.getpid())
    p = Pool(4)
    for i in range(5):
        p.apply_async(long_time_task, args=(i,))
    print("Waiting for all subprocesses done...")
    p.close()
    p.join()
    print("All subprocesses done.")


def write(q: Queue) -> None:
    """
    Write data to a queue.
    """
    print(f"Process[{os.getpid()}] start to write...")
    for v in ["A", "B", "C"]:
        q.put(v)
        print(f"Put {v} to queue...")
        time.sleep(random.random())
    print(f"Process[{os.getpid()}] to write done.")


def read(q: Queue) -> None:
    """
    Read data from a queue.
    """
    print(f"Process[{os.getpid()}] start to read...")
    while True:
        value = q.get(True)
        print(f"Get {value} from queue...")


def test_communication_among_processes():
    """
    Test communication among processes.
    """
    q = Queue()
    pw = Process(target=write, args=(q,))
    pr = Process(target=read, args=(q,))
    pw.start()
    pr.start()
    pw.join()
    pr.terminate()


if __name__ == "__main__":
    # test_multiprocess()
    test_communication_among_processes()
