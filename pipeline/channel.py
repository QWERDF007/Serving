import os
import queue
import sys
import enum
import threading
import multiprocessing
import queue as Queue
import time
import copy
from time import time as _time

import numpy as np

from pipeline.error_catch import CustomExceptionCode as ChannelDataErrCode
from utils.logger import get_logger, set_logger_level

_LOGGER = get_logger()


class ChannelDataType(enum.Enum):
    """
    Channel data type
    """
    DICT = 0
    CHANNEL_NPDATA = 1
    ERROR = 2


class ChannelData(object):
    """
    有几种方式使用 ChannelData：
    1. ChannelData(ChannelDataType.CHANNEL_NPDATA.value, npdata, data_id, log_id)
    2. ChannelData(ChannelDataType.DICT.value, dictdata, data_id, log_id)
    3. ChannelData(error_code, error_info, prod_error_code, prod_error_info, data_id, log_id)

    Protocol buffers 不能序列化:
    https://stackoverflow.com/questions/55344376/how-to-import-protobuf-module
    """

    def __init__(self,
                 datatype=None,
                 npdata=None,
                 dictdata=None,
                 data_id=None,
                 log_id=None,
                 error_code=None,
                 error_info=None,
                 prod_error_code=None,
                 prod_error_info=None,
                 client_need_profile=False):

        if error_code is not None or prod_error_code is not None:
            if data_id is None or error_info is None:
                _LOGGER.critical("Failed to generate ChannelData: data_id and error_info cannot be None")
                os._exit(-1)
            datatype = ChannelDataType.ERROR.value
        else:
            if datatype == ChannelDataType.CHANNEL_NPDATA.value:
                error_code, error_info = ChannelData.check_npdata(npdata)
                if error_code != ChannelDataErrCode.OK.value:
                    datatype = ChannelDataType.ERROR.value
                    _LOGGER.error("(data_id={} log_id={}) {}".format(data_id, log_id, error_info))
            elif datatype == ChannelDataType.DICT.value:
                error_code, error_info = ChannelData.check_dictdata(dictdata)
                if error_code != ChannelDataErrCode.OK.value:
                    datatype = ChannelDataType.ERROR.value
                    _LOGGER.error("(data_id={} log_id={}) {}".format(data_id, log_id, error_info))
            else:
                _LOGGER.critical("(data_id={} log_id={}) datatype not match".format(data_id, log_id))
                os._exit(-1)

        self.datatype = datatype
        self.npdata = npdata
        self.dictdata = dictdata
        self.id = data_id
        self.log_id = log_id
        self.error_code = error_code
        self.error_info = error_info
        self.prod_error_code = prod_error_code
        self.prod_error_info = prod_error_info
        self.client_need_profile = client_need_profile
        self.profile_data_set = set()

    def get_size(self):
        size = 0
        if isinstance(self.dictdata, dict):
            for k in self.dictdata:
                size += sys.getsizeof(self.dictdata[k]) + sys.getsizeof(k)
        if isinstance(self.npdata, dict):
            for k in self.npdata:
                size += sys.getsizeof(self.npdata[k]) + sys.getsizeof(k)
        return size

    def add_profile(self, profile_set):
        if self.client_need_profile is False:
            self.client_need_profile = True
        self.profile_data_set |= profile_set

    @staticmethod
    def check_dictdata(dictdata):
        error_code = ChannelDataErrCode.OK.value
        error_info = None
        if isinstance(dictdata, list):
            # batch data
            for sample in dictdata:
                if not isinstance(sample, dict):
                    error_code = ChannelDataErrCode.TYPE_ERROR.value
                    error_info = "Failed to check data: the type of data must be dict, but get {}.".format(type(sample))
                    break
        elif not isinstance(dictdata, dict):
            # batch size = 1
            error_code = ChannelDataErrCode.TYPE_ERROR.value
            error_info = "Failed to check data: the type of data must be dict, but get {}.".format(type(dictdata))
        return error_code, error_info

    @staticmethod
    def check_batch_npdata(batch):
        error_code = ChannelDataErrCode.OK.value
        error_info = None
        for npdata in batch:
            error_code, error_info = ChannelData.check_npdata(npdata)
            if error_code != ChannelDataErrCode.OK.value:
                break
        return error_code, error_info

    @staticmethod
    def check_npdata(npdata):
        error_code = ChannelDataErrCode.OK.value
        error_info = None
        if isinstance(npdata, list):
            # batch data
            for sample in npdata:
                if not isinstance(sample, dict):
                    error_code = ChannelDataErrCode.TYPE_ERROR.value
                    error_info = "Failed to check data: the value of data must be dict, but get {}.".format(
                        type(sample))
                    break
                for _, value in sample.items():
                    if not isinstance(value, np.ndarray):
                        error_code = ChannelDataErrCode.TYPE_ERROR.value
                        error_info = "Failed to check data: the value of data must be np.ndarray, but get {}.".format(
                            type(value))
                        return error_code, error_info
        elif isinstance(npdata, dict):
            # batch_size = 1
            for _, value in npdata.items():
                if not isinstance(value, np.ndarray) and not (isinstance(value, list) and isinstance(value[0], str)):
                    error_code = ChannelDataErrCode.TYPE_ERROR.value
                    error_info = "Failed to check data: the value of data must be np.ndarray, but get {}.".format(
                        type(value))
                    break
        else:
            error_code = ChannelDataErrCode.TYPE_ERROR.value
            error_info = "Failed to check data: the value of data must be dict, but get {}.".format(type(npdata))
        return error_code, error_info

    def parse(self):
        feed = None
        if self.datatype == ChannelDataType.CHANNEL_NPDATA.value:
            # return narray
            feed = self.npdata
        elif self.datatype == ChannelDataType.DICT.value:
            # return dict
            feed = self.dictdata
        else:
            _LOGGER.critical("Failed to parse channeldata: error type({}) in datatype.".format(self.datatype))
            os._exit(-1)
        return feed

    def __cmp__(self, other):
        if self.id < other.id:
            return -1
        elif self.id == other.id:
            return 0
        else:
            return 1

    def get_all_data(self):
        return "type[{}], error_code[{}], data_id[{}], log_id[{}], dict_size[{}]".format(
            ChannelDataType(self.datatype).name, self.error_code, self.id, self.log_id, self.get_size())


class ProcessChannel(object):
    """
    (进程版本) Channel 用于在 ops 间交流

    1. 支持多个不同 Op feed data (多个生产者)，不同类型数据通过数据 ID 打包
    2. 支持多个不同 Op fetch data (多个消费者)，只有当所有类型 Ops 获取到相同 ID 的数据，数据才会被弹出；
       相同类型 Op 不会得相同 ID 的数据
    3. 函数前端支持超时参数来使用自动批处理

    注意：
    1. channel 中的数据的 ID 必须不同
    2. 函数 add_producer() 和 add_consumer() 不是线程安全的，只能在初始化期间调用

    Channel 中有两个缓冲区和一个队列:

        op_A \                                           / op_D
        op_B - a. input_buf -> b. queue -> c. output_buf - op_E
        op_C /

    a. 在输入缓冲中，多个前驱 Ops 的输入按数据 ID 打包
    b. 打包后的数据将在队列中存储
    c. 为了支持多个后继 Ops 检索数据，输出缓冲维护从队列中获取数据
    """

    def __init__(self,
                 manager,
                 name=None,
                 maxsize=0,
                 channel_recv_first_arrive=False):
        # 对于队列多进程：在放入一个对象到空队列后，在队列的方法：`~Queue.empty` 之前可能会有一个无穷小的延迟
        self._que = manager.PriorityQueue(maxsize=maxsize)
        self._maxsize = maxsize
        self.name = name
        self._stop = manager.Value('i', 0)
        self._cv = multiprocessing.Condition()

        self._producers = []
        self._pushed_producer_count = manager.dict()  # {data_id: count}
        self._input_buf = manager.dict()  # {data_id: {op_name: data}}

        self._reset_max_cursor = 1000000000000000000
        self._consumer_cursors = manager.dict()  # {op_name: cursor}
        self._cursor_count = manager.dict()  # {cursor: count}
        self._base_cursor = manager.Value('i', 0)
        self._output_buf = manager.list()

        self._cur_max_dataid = manager.Value('i', -1)
        self._channel_recv_first_arrive = channel_recv_first_arrive

    def get_maxsize(self):
        return self._maxsize

    def size(self):
        return self._que.qsize()

    def get_producers(self):
        return self._producers

    def get_consumers(self):
        return self._consumer_cursors.keys()

    def _log(self, info_str):
        """ 输出格式化 """
        return "[{}] {}".format(self.name, info_str)

    def add_producer(self, op_name):
        """ 非线程安全，只能在初始化期间被调用 """
        if op_name in self._producers:
            _LOGGER.critical(self._log(f"Failed to add producer: producer({op_name}) is already in channel"))
            os._exit(-1)
        self._producers.append(op_name)
        _LOGGER.debug(self._log("Succ add a producer: {}".format(op_name)))

    def add_consumer(self, op_name):
        """ 非线程安全，只能在初始化期间被调用 """
        if op_name in self._consumer_cursors:
            _LOGGER.critical(self._log(f"Failed to add consumer: consumer({op_name}) is already in channel"))
            os._exit(-1)
        self._consumer_cursors[op_name] = 0

        if self._cursor_count.get(0) is None:
            self._cursor_count[0] = 0
        self._cursor_count[0] += 1
        _LOGGER.debug(self._log("Succ add a consumer: {}".format(op_name)))

    def push(self, channeldata: ChannelData, op_name: str):
        _LOGGER.debug(self._log(
            f"(data_id={channeldata.id} log_id={channeldata.log_id}) Op({op_name}) Enter channel:push "
            f"producers:{len(self._producers)}, time:{time.time()}"
        ))
        if len(self._producers) == 0:
            _LOGGER.critical(self._log(
                f"(data_id={channeldata.id} log_id={channeldata.log_id}) Op({op_name}) Failed to push data: "
                f"expected number of producers to be greater than 0, but the it is 0."
            ))
            os._exit(-1)
        elif len(self._producers) == 1:
            start_time = _time()
            with self._cv:
                enter_cv_time = _time()
                push_que_time = enter_cv_time
                while self._stop.value == 0:
                    try:
                        self._que.put((channeldata.id, {op_name: channeldata}), timeout=0)
                        push_que_time = _time()
                        break
                    except Queue.Full:
                        self._cv.wait()
                if self._stop.value == 1:
                    raise ChannelStopError()
                self._cv.notify_all()
                notify_all_time = _time()
                _LOGGER.debug(
                    "(data_id={}) Op({}) channel push cost! enter_cv:{} ms, push_que:{} ms, "
                    "notify:{} ms, data_size:{}, time:{}".format(
                        channeldata.id, op_name,
                        (enter_cv_time - start_time) * 1000,
                        (push_que_time - enter_cv_time) * 1000,
                        (notify_all_time - push_que_time) * 1000,
                        channeldata.get_size(), time.time()))
            _LOGGER.debug(self._log("(data_id={} log_id={}) Op({}) Pushed data into internal queue.".format(
                channeldata.id, channeldata.log_id, op_name)))
            return True
        elif self._channel_recv_first_arrive:
            start_time = _time()
            with self._cv:
                _LOGGER.debug(
                    "(data_id={}) Op({}) Channel({}) enter channel_recv_first_arrive. _cur_max_dataid:{}".format(
                        channeldata.id, op_name, self.name, self._cur_max_dataid.value))
                if channeldata.id > self._cur_max_dataid.value:
                    enter_cv_time = _time()
                    push_que_time = enter_cv_time
                    while self._stop.value == 0:
                        try:
                            self._que.put((channeldata.id, {op_name: channeldata}), timeout=0)
                            push_que_time = _time()
                            self._cur_max_dataid.value = channeldata.id
                            break
                        except Queue.Full:
                            self._cv.wait()
                    if self._stop.value == 1:
                        raise ChannelStopError()
                    self._cv.notify_all()
                    notify_all_time = _time()
                    _LOGGER.debug(
                        "(data_id={}) Op({}) channel push cost! enter_cv:{} ms, push_que:{} ms, "
                        "notify:{} ms, data_size:{}, time:{}".format(
                            channeldata.id, op_name,
                            (enter_cv_time - start_time) * 1000,
                            (push_que_time - enter_cv_time) * 1000,
                            (notify_all_time - push_que_time) * 1000,
                            channeldata.get_size(), time.time()))
                else:
                    # log and drop it
                    _LOGGER.debug("(data_id={}) Op({}) send data is dropped! cur_max_dataid:{}".format(
                        channeldata.id, op_name, self._cur_max_dataid.value))
            return True
        elif op_name is None:
            _LOGGER.critical(self._log(
                "(data_id={} log_id={}) Op({}) Failed to push data: there are multiple producers, "
                "so op_name cannot be None.".format(
                    channeldata.id, channeldata.log_id, op_name)))
            os._exit(-1)

        producer_num = len(self._producers)
        data_id = channeldata.id
        log_id = channeldata.log_id
        put_data = None
        with self._cv:
            if data_id not in self._input_buf:
                self._input_buf[data_id] = {name: None for name in self._producers}
                self._pushed_producer_count[data_id] = 0
            # see: https://docs.python.org/3.6/library/multiprocessing.html?highlight=multiprocess#proxy-objects
            # self._input_buf[data_id][op_name] = channeldata
            # 按照 data_id 将数据对齐？
            tmp_input_buf = self._input_buf[data_id]
            tmp_input_buf[op_name] = channeldata
            self._input_buf[data_id] = tmp_input_buf

            if self._pushed_producer_count[data_id] + 1 == producer_num:
                put_data = self._input_buf[data_id]
                self._input_buf.pop(data_id)
                self._pushed_producer_count.pop(data_id)
            else:
                self._pushed_producer_count[data_id] += 1

            if put_data is None:
                _LOGGER.debug(self._log("(data_id={} log_id={}) Op({}) Pushed data into input_buffer.".format(
                    data_id, log_id, op_name)))
            else:
                while self._stop.value == 0:
                    try:
                        self._que.put((data_id, put_data), timeout=0)
                        break
                    except Queue.Full:
                        self._cv.wait()
                if self._stop.value == 1:
                    raise ChannelStopError()

                _LOGGER.debug(
                    self._log("(data_id={} log_id={}) Op({}) Pushed data into internal_queue. time:{}".format(
                        data_id, log_id, op_name, time.time())))
            self._cv.notify_all()
        return True

    def front(self, op_name=None, timeout=None):
        _LOGGER.debug(self._log("Op({}) Getting data[?]; timeout(s)={}".format(op_name, timeout)))

        endtime = None
        if timeout is not None:
            if timeout <= 0:
                timeout = None
            else:
                endtime = _time() + timeout

        if len(self._consumer_cursors) == 0:
            _LOGGER.critical(
                self._log(
                    "Op({}) Failed to get data: expected number of consumers to be "
                    "greater than 0, but the it is 0.".format(op_name)))
            os._exit(-1)
        elif len(self._consumer_cursors) == 1:
            resp = None
            time_1 = int(round(_time() * 1000000))
            time_2 = time_1
            time_3 = time_2
            with self._cv:
                time_2 = int(round(_time() * 1000000))
                while self._stop.value == 0 and resp is None:
                    try:
                        resp = self._que.get(timeout=0)[1]
                        time_3 = int(round(_time() * 1000000))
                        break
                    except Queue.Empty:
                        if timeout is not None:
                            remaining = endtime - _time()
                            if remaining <= 0.0:
                                _LOGGER.debug(self._log("Op({}) Failed to get data: timeout".format(op_name)))
                                raise ChannelTimeoutError()
                            self._cv.wait(remaining)
                        else:
                            self._cv.wait()
                if self._stop.value == 1:
                    raise ChannelStopError()
            key = list(resp.keys())[0]
            data_id = resp[key].id
            _LOGGER.debug("(data_id={}) op({}) front cost enter_cv:{} ms, queue_get:{} ms, time:{}".format(
                data_id, op_name, (time_2 - time_1) / 1000.0, (time_3 - time_2) / 1000.0, time.time()))
            if resp is not None:
                list_values = list(resp.values())
                _LOGGER.debug(self._log("(data_id={} log_id={}) Op({}) Got data".format(
                    list_values[0].id, list_values[0].log_id, op_name)))
            return resp
        elif op_name is None:
            _LOGGER.critical(
                self._log(
                    "Op({}) Failed to get data: there are multiple consumers, "
                    "so op_name cannot be None.".format(op_name)))
            os._exit(-1)

        # 在输出缓冲中，不同的 Ops (根据 op_name) 有不同的游标。此外，有一个 base_cursor。
        # 它们的不同之处是当前时刻对应的 Op 获取的 data_idx = consumer_cursor - base_cursor
        #
        #            base_cursor    consumer_B_cursor (data_idx: 3)
        #                 |                       |
        # output_buf: | data0 | data1 | data2 | data3 |
        #                 |
        #   consumer_A_cursor (data_idx: 0)
        with self._cv:
            # 当当前 Op 需要的数据不在输出缓冲时，必须从队列中获取数据并添加到输出缓冲
            while self._stop.value == 0 and self._consumer_cursors[op_name] - self._base_cursor.value >= len(
                    self._output_buf):
                try:
                    channeldata = self._que.get(timeout=0)[1]
                    self._output_buf.append(channeldata)
                    list_values = list(channeldata.values())
                    _LOGGER.debug(
                        self._log("(data_id={} log_id={}) Op({}) Pop ready item into output_buffer, time:{}".format(
                            list_values[0].id, list_values[0].log_id, op_name, time.time())))
                    break
                except Queue.Empty:
                    if timeout is not None:
                        remaining = endtime - _time()
                        if remaining <= 0.0:
                            _LOGGER.debug(self._log("Op({}) Failed to get data: timeout".format(op_name)))
                            raise ChannelTimeoutError()
                        self._cv.wait(remaining)
                    else:
                        self._cv.wait()
            if self._stop.value == 1:
                raise ChannelStopError()

            time_1 = int(round(_time() * 1000000))
            consumer_cursor = self._consumer_cursors[op_name]
            base_cursor = self._base_cursor.value
            data_idx = consumer_cursor - base_cursor
            resp = self._output_buf[data_idx]

            self._cursor_count[consumer_cursor] -= 1
            if consumer_cursor == base_cursor and self._cursor_count[consumer_cursor] == 0:
                # 当所有不同的 Ops 得到 data_idx 指向的数据时，将其从输出缓冲中弹出
                self._cursor_count.pop(consumer_cursor)
                self._output_buf.pop(0)
                self._base_cursor.value += 1

                # 避免游标溢出
                if self._base_cursor.value >= self._reset_max_cursor:
                    _LOGGER.info(self._log("Reset cursor in Channel"))
                    self._base_cursor.value -= self._reset_max_cursor
                    for name in self._consumer_cursors.keys():
                        self._consumer_cursors[name] -= self._reset_max_cursor
                    cursor_count_tmp = {
                        cursor - self._reset_max_cursor: count for cursor, count in self._cursor_count.copy().items()
                    }
                    self._cursor_count.clear()
                    for cursor, count in cursor_count_tmp.items():
                        self._cursor_count[cursor] = count

            self._consumer_cursors[op_name] += 1
            new_consumer_cursor = self._consumer_cursors[op_name]
            if self._cursor_count.get(new_consumer_cursor) is None:
                self._cursor_count[new_consumer_cursor] = 0
            self._cursor_count[new_consumer_cursor] += 1

            self._cv.notify_all()
            time_2 = int(round(_time() * 1000000))
            # _LOGGER.debug("self._cv logic cost:{}".format(time_2 - time_1))

        if resp is not None:
            list_values = list(resp.values())
            _LOGGER.debug(
                self._log("(data_id={} log_id={}) Op({}) Got data from output_buffer, time:{}".format(
                    list_values[0].id, list_values[0].log_id, op_name, time.time())))
        return resp

    def stop(self):
        _LOGGER.info(self._log("stop."))
        self._stop.value = 1
        with self._cv:
            self._cv.notify_all()


class ThreadChannel(Queue.PriorityQueue):
    """
    (线程版本) Channel 用于在 ops 间交流

    1. 支持多个不同 Op feed data (多个生产者)，不同类型数据通过数据 ID 打包
    2. 支持多个不同 Op fetch data (多个消费者)，只有当所有类型 Ops 获取到相同 ID 的数据，数据才会被弹出；
       相同类型 Op 不会得相同 ID 的数据
    3. 函数前端支持超时参数来使用自动批处理

    注意：
    1. channel 中的数据的 ID 必须不同
    2. 函数 add_producer() 和 add_consumer() 不是线程安全的，只能在初始化期间调用

    Channel 中有两个缓冲区和一个队列:

        op_A \                                           / op_D
        op_B - a. input_buf -> b. queue -> c. output_buf - op_E
        op_C /

    a. 在输入缓冲中，多个前驱 Ops 的输入按数据 ID 打包
    b. 打包后的数据将在队列中存储
    c. 为了支持多个后继 Ops 检索数据，输出缓冲维护从队列中获取数据
    """

    def __init__(self,
                 name=None,
                 maxsize=-1,
                 channel_recv_first_arrive=False):
        Queue.Queue.__init__(self, maxsize=maxsize)
        self._maxsize = maxsize
        self.name = name
        self._stop = False
        self._cv = threading.Condition()

        self._producers = []
        self._pushed_producer_count = {}  # {data_id: count}
        self._input_buf = {}  # {data_id: {op_name: data}}

        self._reset_max_cursor = 1000000000000000000
        self._consumer_cursors = {}  # {op_name: idx}
        self._cursor_count = {}  # {cursor: count}
        self._base_cursor = 0
        self._output_buf = []

        self._channel_recv_first_arrive = channel_recv_first_arrive
        self._cur_max_dataid = -1

    def get_maxsize(self):
        return self._maxsize

    def size(self):
        return self.qsize()

    def get_producers(self):
        return self._producers

    def get_consumers(self):
        return self._consumer_cursors.keys()

    def _log(self, info_str):
        """输出格式化"""
        return "[{}] {}".format(self.name, info_str)

    def add_producer(self, op_name):
        """ 非线程安全，只能在初始化期间被调用 """
        if op_name in self._producers:
            _LOGGER.critical(self._log("Failed to add producer: producer({}) is already in channel".format(op_name)))
            os._exit(-1)
        self._producers.append(op_name)
        _LOGGER.debug(self._log("Succ add a producer: {}".format(op_name)))

    def add_consumer(self, op_name):
        """ 非线程安全，只能在初始化期间被调用 """
        if op_name in self._consumer_cursors:
            _LOGGER.critical(self._log("Failed to add consumer: consumer({}) is already in channel".format(op_name)))
            os._exit(-1)
        self._consumer_cursors[op_name] = 0

        if self._cursor_count.get(0) is None:
            self._cursor_count[0] = 0
        self._cursor_count[0] += 1
        _LOGGER.debug(self._log("Succ add a consumer: {}".format(op_name)))

    def push(self, channeldata: ChannelData, op_name=None):
        _LOGGER.debug(self._log("(data_id={} log_id={}) Op({}) Pushing data".format(
            channeldata.id, channeldata.log_id, op_name)))

        if len(self._producers) == 0:
            _LOGGER.critical(self._log(
                "(data_id={} log_id={}) Op({}) Failed to push data: expected number of "
                "producers to be greater than 0, but the it is 0.".format(
                    channeldata.id, channeldata.log_id, op_name)))
            os._exit(-1)
        elif len(self._producers) == 1:
            with self._cv:
                while not self._stop:
                    try:
                        self.put((channeldata.id, {op_name: channeldata}), timeout=0)
                        break
                    except Queue.Full:
                        self._cv.wait()
                if self._stop:
                    raise ChannelStopError()
                self._cv.notify_all()
            _LOGGER.debug(self._log("(data_id={} log_id={}) Op({}) Pushed data into internal_queue.".format(
                channeldata.id, channeldata.log_id, op_name)))
            return True
        elif self._channel_recv_first_arrive:
            with self._cv:
                if channeldata.id > self._cur_max_dataid:
                    while not self._stop:
                        try:
                            self.put((channeldata.id, {op_name: channeldata}), timeout=0)
                            self._cur_max_dataid = channeldata.id
                        except queue.Full:
                            self._cv.wait()
                    if self._stop:
                        raise ChannelStopError()
                    self._cv.notify_all()
                else:
                    _LOGGER.debug("(data_id={}) Op({}) send data is dropped! cur_max_dataid:{}".format(
                        channeldata.id, op_name, self._cur_max_dataid))
            return True
        elif op_name is None:
            _LOGGER.critical(self._log(
                "(data_id={} log_id={}) Op({}) Failed to push data: there are multiple"
                " producers, so op_name cannot be None.".format(
                    channeldata.id, channeldata.log_id, op_name)))
            os._exit(-1)

        producer_num = len(self._producers)
        data_id = channeldata.id
        log_id = channeldata.log_id
        put_data = None
        with self._cv:
            if data_id not in self._input_buf:
                self._input_buf[data_id] = {name: None for name in self._producers}
                self._pushed_producer_count[data_id] = 0
            self._input_buf[data_id][op_name] = channeldata
            if self._pushed_producer_count[data_id] + 1 == producer_num:
                put_data = self._input_buf[data_id]
                self._input_buf.pop(data_id)
                self._pushed_producer_count.pop(data_id)
            else:
                self._pushed_producer_count[data_id] += 1

            if put_data is None:
                _LOGGER.debug(self._log("(data_id={} log_id={}) Op({}) Pushed data into input_buffer.".format(
                    data_id, log_id, op_name)))
            else:
                while not self._stop:
                    try:
                        self.put((data_id, put_data), timeout=0)
                        break
                    except Queue.Empty:  # TODO queue.put 会抛出 Queue.Empty ?
                        self._cv.wait()
                if self._stop:
                    raise ChannelStopError()

                _LOGGER.debug(self._log("(data_id={} log_id={}) Op({}) Pushed data into internal_queue.".format(
                    data_id, log_id, op_name)))
            self._cv.notify_all()
        return True

    def front(self, op_name=None, timeout=None):
        _LOGGER.debug(self._log("Op({}) Getting data[?]; timeout(s)={}".format(op_name, timeout)))

        endtime = None
        if timeout is not None:
            if timeout <= 0:
                timeout = None
            else:
                endtime = _time() + timeout

        if len(self._consumer_cursors) == 0:
            _LOGGER.critical(self._log(
                "Op({}) Failed to get data: expected number of consumers to be "
                "greater than 0, but the it is 0.".format(op_name)))
            os._exit(-1)
        elif len(self._consumer_cursors) == 1:
            resp = None
            with self._cv:
                while not self._stop and resp is None:
                    try:
                        resp = self.get(timeout=0)[1]
                        break
                    except Queue.Empty:
                        if timeout is not None:
                            remaining = endtime - _time()
                            if remaining <= 0.0:
                                _LOGGER.debug(self._log("Op({}) Failed to get data: timeout".format(op_name)))
                                raise ChannelTimeoutError()
                            self._cv.wait(remaining)
                        else:
                            self._cv.wait()
                if self._stop:
                    raise ChannelStopError()

            if resp is not None:
                list_values = list(resp.values())
                _LOGGER.debug(self._log("(data_id={} log_id={}) Op({}) Got data".format(
                    list_values[0].id, list_values[0].log_id, op_name)))
            return resp
        elif op_name is None:
            _LOGGER.critical(self._log(
                "Op({}) Failed to get data: there are multiple "
                "consumers, so op_name cannot be None.".format(op_name)))
            os._exit(-1)

        # 在输出缓冲中，不同的 Ops (根据 op_name) 有不同的游标。此外，有一个 base_cursor。
        # 它们的不同之处是当前时刻对应的 Op 获取的 data_idx = consumer_cursor - base_cursor
        #
        #            base_cursor    consumer_B_cursor (data_idx: 3)
        #                 |                       |
        # output_buf: | data0 | data1 | data2 | data3 |
        #                 |
        #   consumer_A_cursor (data_idx: 0)
        with self._cv:
            # 当当前 Op 需要的数据不在输出缓冲时，必须从队列中获取数据并添加到输出缓冲
            while not self._stop and self._consumer_cursors[op_name] - self._base_cursor >= len(self._output_buf):
                try:
                    channeldata = self.get(timeout=0)[1]
                    self._output_buf.append(channeldata)
                    list_values = list(channeldata.values())
                    _LOGGER.debug(self._log("(data_id={} log_id={}) Op({}) Pop ready item into output_buffer".format(
                        list_values[0].id, list_values[0].log_id, op_name)))
                    break
                except Queue.Empty:
                    if timeout is not None:
                        remaining = endtime - _time()
                        if remaining <= 0.0:
                            _LOGGER.debug(self._log("Op({}) Failed to get data: timeout".format(op_name)))
                            raise ChannelTimeoutError()
                        self._cv.wait(remaining)
                    else:
                        self._cv.wait()
            if self._stop:
                raise ChannelStopError()

            consumer_cursor = self._consumer_cursors[op_name]
            base_cursor = self._base_cursor
            data_idx = consumer_cursor - base_cursor

            resp = None

            self._cursor_count[consumer_cursor] -= 1
            if consumer_cursor == base_cursor and self._cursor_count[consumer_cursor] == 0:
                # 当所有不同的 Ops 得到 data_idx 指向的数据时，将其从输出缓冲中弹出
                self._cursor_count.pop(consumer_cursor)
                resp = self._output_buf.pop(0)
                self._base_cursor += 1

                # 避免游标溢出
                if self._base_cursor >= self._reset_max_cursor:
                    _LOGGER.info(self._log("Reset cursor in Channel"))
                    self._base_cursor -= self._reset_max_cursor
                    for name in self._consumer_cursors:
                        self._consumer_cursors[name] -= self._reset_max_cursor
                    self._cursor_count = {
                        cursor - self._reset_max_cursor: count for cursor, count in self._cursor_count.items()
                    }
            else:
                resp = copy.deepcopy(self._output_buf[data_idx])

            self._consumer_cursors[op_name] += 1
            new_consumer_cursor = self._consumer_cursors[op_name]
            if self._cursor_count.get(new_consumer_cursor) is None:
                self._cursor_count[new_consumer_cursor] = 0
            self._cursor_count[new_consumer_cursor] += 1

            self._cv.notify_all()

        if resp is not None:
            list_values = list(resp.values())
            _LOGGER.debug(self._log("(data_id={} log_id={}) Op({}) Got data from output_buffer".format(
                list_values[0].id, list_values[0].log_id, op_name)))

        return resp

    def stop(self):
        _LOGGER.info(self._log("stop."))
        self._stop = True
        with self._cv:
            self._cv.notify_all()


class ChannelTimeoutError(RuntimeError):
    def __init__(self):
        pass


class ChannelStopError(RuntimeError):
    def __init__(self):
        pass
