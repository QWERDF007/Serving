import socket
from contextlib import closing


def port_is_available(port):
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.settimeout(2)
        result = sock.connect_ex(('127.0.0.1', port))
    if result != 0:
        return True
    else:
        return False
