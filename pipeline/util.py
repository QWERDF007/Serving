import threading
import multiprocessing
from queue import PriorityQueue

from utils.logger import get_logger
from utils.network_util import port_is_available

_LOGGER = get_logger("Serving")


class AvailablePortGenerator(object):
    def __init__(self, start_port=12000):
        self._curr_port = start_port

    def next(self):
        while not port_is_available(self._curr_port):
            self._curr_port += 1
        self._curr_port += 1
        return self._curr_port - 1


_AvailablePortGenerator = AvailablePortGenerator()


def GetAvailablePortGenerator():
    return _AvailablePortGenerator


class NameGenerator(object):
    # use unsafe-id-generator
    def __init__(self, prefix):
        self._idx = -1
        self._prefix = prefix
        self._id_generator = UnsafeIdGenerator(1000000000000000000)

    def next(self):
        next_id = self._id_generator.next()
        return "{}{}".format(self._prefix, next_id)


class UnsafeIdGenerator(object):
    def __init__(self, max_id, base_counter=0, step=1):
        self._base_counter = base_counter
        self._counter = self._base_counter
        self._step = step
        self._max_id = max_id  # for reset

    def next(self):
        if self._counter >= self._max_id:
            self._counter = self._base_counter
            _LOGGER.info("Reset Id: {}".format(self._counter))
        next_id = self._counter
        self._counter += self._step
        return next_id


class ThreadIdGenerator(UnsafeIdGenerator):
    def __init__(self, max_id, base_counter=0, step=1, lock=None):
        # if you want to use your lock, you may need to use Reentrant-Lock
        self._lock = lock
        if self._lock is None:
            self._lock = threading.Lock()
        super(ThreadIdGenerator, self).__init__(max_id, base_counter, step)

    def next(self):
        next_id = None
        with self._lock:
            if self._counter >= self._max_id:
                self._counter = self._base_counter
                _LOGGER.info("Reset Id: {}".format(self._counter))
            next_id = self._counter
            self._counter += self._step
        return next_id


class ProcessIdGenerator(UnsafeIdGenerator):
    def __init__(self, max_id, base_counter=0, step=1, lock=None):
        # if you want to use your lock, you may need to use Reentrant-Lock
        self._lock = lock
        if self._lock is None:
            self._lock = multiprocessing.Lock()
        self._base_counter = base_counter
        self._counter = multiprocessing.Manager().Value('i', 0)
        self._step = step
        self._max_id = max_id

    def next(self):
        next_id = None
        with self._lock:
            if self._counter.value >= self._max_id:
                self._counter.value = self._base_counter
                _LOGGER.info("Reset Id: {}".format(self._counter.value))
            next_id = self._counter.value
            self._counter.value += self._step
        return next_id


def PipelineProcSyncManager():
    """
    add PriorityQueue into SyncManager, see more:
    https://stackoverflow.com/questions/25324560/strange-queue-priorityqueue-behaviour-with-multiprocessing-in-python-2-7-6?answertab=active#tab-top
    """

    class PipelineManager(multiprocessing.managers.SyncManager):
        pass

    PipelineManager.register("PriorityQueue", PriorityQueue)
    m = PipelineManager()
    m.start()
    return m
