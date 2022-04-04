import os
import sys
import copy
import time
import threading
import queue as Queue
import multiprocessing

from utils.logger import get_logger
from utils.time_util import timestamp as _timestamp

_LOGGER = get_logger("Serving")
_is_profile = int(os.environ.get('FLAGS_profile_pipeline', 0))


class PerformanceTracer(object):
    """
    性能追踪
    """

    def __init__(self, is_thread_mode, interval_s, server_worker_num):
        self._is_thread_mode = is_thread_mode
        if is_thread_mode:
            # 线程模式中的 Channel 不能跨进程访问，所以当使用线程模式时，PerformanceTracer 也是线程模式。
            # 性能可能会受到 GIL 影响。
            self._data_buffer = Queue.Queue()
        else:
            self._data_buffer = multiprocessing.Manager().Queue()
        self._interval_s = interval_s
        self._thread = None
        self._process = None
        self._channels = []
        # channel 中的数据大小不会超过 server_worker_num
        self._server_worker_num = server_worker_num
        self.profile_dict = {}
        self._enable_dict = False

    def data_buffer(self):
        return self._data_buffer

    def start(self):
        if self._is_thread_mode:
            self._thread = threading.Thread(target=self._trace_func, args=(self._channels,))
            self._thread.daemon = True
            self._thread.start()
        else:
            self._process = multiprocessing.Process(target=self._trace_func, args=(self._channels,))
            self._process.daemon = True
            self._process.start()

    def set_channels(self, channels):
        self._channels = channels

    def set_enable_dict(self, enable):
        self._enable_dict = enable

    def _trace_func(self, channels):
        all_actions = ["in", "prep", "midp", "postp", "out"]
        calcu_actions = ["prep", "midp", "postp"]
        while True:
            op_cost = {}
            err_request = []
            err_count = 0

            _LOGGER.info("==================== TRACER ======================")
            # op
            while True:
                try:
                    item = self._data_buffer.get_nowait()
                    name = item["name"]
                    actions = item["actions"]

                    if name == "DAG":
                        succ = item["succ"]
                        req_id = item["id"]
                        if not succ:
                            err_count += 1
                            err_request.append(req_id)

                    if name not in op_cost:
                        op_cost[name] = {}

                    for action, cost in actions.items():
                        if action not in op_cost[name]:
                            op_cost[name][action] = []
                        op_cost[name][action].append(cost)
                except Queue.Empty:
                    break

            if len(op_cost) != 0:
                for name in op_cost:
                    tot_cost, calcu_cost = 0.0, 0.0
                    count = 0
                    for action, costs in op_cost[name].items():
                        op_cost[name][action] = sum(costs) / (1e3 * len(costs))
                        tot_cost += op_cost[name][action]
                        if action == "midp":
                            count = len(costs)
                    if "midp" in op_cost[name].keys():
                        op_cost[name]['count'] = count
                    if name != "DAG":
                        _LOGGER.info("Op({}):".format(name))

                        for action in all_actions:
                            if action in op_cost[name]:
                                _LOGGER.info("\t{}[{} ms]".format(
                                    action, op_cost[name][action]))
                        for action in calcu_actions:
                            if action in op_cost[name]:
                                calcu_cost += op_cost[name][action]
                        _LOGGER.info("\tidle[{}]".format(1 - 1.0 * calcu_cost /
                                                         tot_cost))
            self.profile_dict = copy.deepcopy(op_cost)

            if "DAG" in op_cost:
                calls = list(op_cost["DAG"].values())
                calls.sort()
                tot = len(calls)
                qps = 1.0 * tot / self._interval_s
                ave_cost = sum(calls) / tot
                latencys = [50, 60, 70, 80, 90, 95, 99]
                _LOGGER.info("DAGExecutor:")
                _LOGGER.info("\tQuery count[{}]".format(tot))
                _LOGGER.info("\tQPS[{} q/s]".format(qps))
                _LOGGER.info("\tSucc[{}]".format(1 - 1.0 * err_count / tot))
                _LOGGER.info("\tError req[{}]".format(", ".join(
                    [str(x) for x in err_request])))
                _LOGGER.info("\tLatency:")
                _LOGGER.info("\t\tave[{} ms]".format(ave_cost))
                for latency in latencys:
                    _LOGGER.info("\t\t.{}[{} ms]".format(latency, calls[int(
                        tot * latency / 100.0)]))
                if _is_profile or self._enable_dict:
                    self.profile_dict["DAG"]["query_count"] = tot
                    self.profile_dict["DAG"]["qps"] = qps
                    self.profile_dict["DAG"]["succ"] = 1 - 1.0 * err_count / tot
                    self.profile_dict["DAG"]["avg"] = ave_cost
                    for latency in latencys:
                        self.profile_dict["DAG"][str(latency)] = calls[int(tot * latency / 100.0)]
            if _is_profile:
                import yaml
                with open("benchmark.log", "w") as fout:
                    yaml.dump(self.profile_dict, fout, default_flow_style=False)
            # channel
            _LOGGER.info("Channel (server worker num[{}]):".format(
                self._server_worker_num))
            for channel in channels:
                _LOGGER.info("\t{}(In: {}, Out: {}) size[{}/{}]".format(
                    channel.name,
                    channel.get_producers(),
                    channel.get_consumers(),
                    channel.size(), channel.get_maxsize()))
            time.sleep(self._interval_s)


class TimeProfiler(object):
    def __init__(self):
        self._pid = os.getpid()
        self._print_head = 'PROFILE\tpid:{}\t'.format(self._pid)
        self._time_record = Queue.Queue()
        self._enable = False
        self._lock = threading.Lock()

    def enable(self, enable):
        self._enable = enable

    def record(self, name_with_tag):
        """
        记录当前时间戳

        Args:
            name_with_tag: name with tag

        Returns:
            当前时间戳 int(round(_time() * 1000000))
        """
        if self._enable is False:
            return
        timestamp = _timestamp()
        name_with_tag = name_with_tag.split("_")
        tag = name_with_tag[-1]
        name = '_'.join(name_with_tag[:-1])
        with self._lock:
            self._time_record.put((name, tag, timestamp))
        return timestamp

    def print_profile(self):
        if self._enable is False:
            return
        sys.stderr.write(self.gen_profile_str())

    def gen_profile_str(self):
        if self._enable is False:
            return
        print_str = self._print_head
        tmp = {}
        with self._lock:
            while not self._time_record.empty():
                name, tag, timestamp = self._time_record.get()
                if name in tmp:
                    ptag, ptimestamp = tmp.pop(name)
                    print_str += "{}_{}:{} ".format(name, ptag, ptimestamp)
                    print_str += "{}_{}:{} ".format(name, tag, timestamp)
                else:
                    tmp[name] = (tag, timestamp)
            print_str = "\n{}\n".format(print_str)
            for name, item in tmp.items():
                tag, timestamp = item
                self._time_record.put((name, tag, timestamp))
            return print_str
