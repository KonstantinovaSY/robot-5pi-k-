import socket
from time import sleep

host = "192.168.2.35"
port = 2001

def create_connection(host, port):
    connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connection.connect((host, port))
    return connection

def send_command(connection, command):
    try:
        connection.sendall(command)
        #data = connection.recv(1024)
        #print(data)
        return True
    except socket.error as e:
        print(f"Ошибка сокета: {e}")
        return False

def close_connection(connection):
    connection.close()

if __name__ == '__main__':
    c = create_connection(host, port)
    forward = b'\xFF\x00\x01\x00\xFF'
    print(send_command(c, forward))
    sleep(1)
    stop = b'\xFF\x00\x00\x00\xFF'
    send_command(c, stop)
    close_

