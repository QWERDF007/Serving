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


def get_local_ip():
    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)
    if ip.startswith('127'):
        st = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            st.connect(('10.255.255.255', 1))
            ip = st.getsockname()[0]
        except Exception:
            ip = "127.0.0.1"
        finally:
            st.close()
    return ip

