import time


def timestamp():
    """
    得到当前时间戳，16位十进制数
    :return:  16 位十进制时间戳
    """
    return int(round(time.time() * 1000000))
