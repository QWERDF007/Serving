import os
import time
import json
import signal
import platform


def pid_is_exist(pid: int):
    """
    尝试通过 pid 杀进程
    Args:
        pid: 将要被杀掉的进程 pid

    Returns:
        如果进程将被杀掉返回 True
    """

    try:
        os.kill(pid, signal.CTRL_C_EVENT)
    except:
        return False
    else:
        return True


def kill_stop_process_by_pid(command: str, pid: int):
    """
    使用不同的信号杀掉 pid 进程组
    Args:
        command: stop -> SIGINT, kill -> SIGKILL
        pid: 将要杀掉的进程 pid

    Returns:

    """
    if not pid_is_exist(pid):
        print("Process [%s]  has been stopped." % pid)
        return
    if platform.system() == "Windows":
        os.kill(pid, signal.SIGINT)
    else:
        try:
            if command == "stop":
                os.killpg(pid, signal.SIGINT)
            elif command == "kill":
                os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            if command == "stop":
                os.kill(pid, signal.SIGINT)
            elif command == "kill":
                os.kill(pid, signal.SIGKILL)


def dump_pid_file(port_list, model):
    """
    Write PID info to file.

    Args:
        port_list(List): PiplineServing includes http_port and rpc_port
                        PaddleServing include one port
        model(str): 'Pipline' for PiplineServing
                    Specific model list for ServingModel

    Returns:
       None
    Examples:
    .. code-block:: python

       dump_pid_file([9494, 10082], 'serve')
    """

    pid = os.getpid()
    if platform.system() == "Windows":
        gid = pid
    else:
        gid = os.getpgid(pid)
    pid_info_list = []
    # filepath = os.path.join(CONF_HOME, "ProcessInfo.json")
    filepath = "ProcessInfo.json"
    if os.path.exists(filepath):
        if os.path.getsize(filepath):
            with open(filepath, "r") as fp:
                pid_info_list = json.load(fp)
                # delete old pid data when new port number is same as old's
                for info in pid_info_list:
                    stored_port = list(info["port"])
                    inter_list = list(set(port_list) & set(stored_port))
                    if inter_list:
                        pid_info_list.remove(info)

    with open(filepath, "w") as fp:
        info = {"pid": gid, "port": port_list, "model": str(model), "start_time": time.time()}
        pid_info_list.append(info)
        json.dump(pid_info_list, fp)
