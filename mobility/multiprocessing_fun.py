import time
import random
from multiprocessing import Process
from multiprocessing import Queue
from multiprocessing import current_process
from multiprocessing import freeze_support
from multiprocessing import cpu_count

import ee

#
# Function run by worker processes
#

def worker(input, output):
    for func, args in iter(input.get, 'STOP'):
        result = calculate(func, args)
        output.put(result)


def calculate(func, args):
    result = func(*args)
    return '%s says that %s%s = %s' % \
        (current_process().name, func.__name__, args, result)


def export(image, out):
    ee_export_image(
        image,
        out,
        30
    )
    return True

def multiprocess(tasks):
    NUMBER_OF_PROCESSES = cpu_count() - 1

    # Create queues
    task_queue = Queue()
    done_queue = Queue()

    # Submit tasks
    for task in tasks:
        task_queue.put(task)

    # Start worker processes
    for i in range(NUMBER_OF_PROCESSES):
        Process(target=worker, args=(task_queue, done_queue)).start()

    # Get and print results
    print('Unordered results:')
    for i in range(len(tasks)):
        print('\t', done_queue.get())

    # Tell child processes to stop
    for i in range(NUMBER_OF_PROCESSES):
        task_queue.put('STOP')
