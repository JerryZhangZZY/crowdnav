"""
Run this on MyAGV
"""

import socket
from pymycobot.myagv import MyAgv

TIME_OUT = 1


def start_server(host='0.0.0.0', port=12345):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)
    print(f"Server listening on {host}:{port}")

    while True:
        try:
            conn, addr = server_socket.accept()
            print(f"Connection from {addr}")
            conn.settimeout(TIME_OUT)

            while True:
                try:
                    data = conn.recv(1024)
                    if not data:
                        break
                    numbers = data.decode('utf-8').split(',')
                    num1, num2, num3 = int(numbers[0]), int(numbers[1]), int(numbers[2])
                    print({num1}, {num2}, {num3})
                    try:
                        MA._mesg(128 + num1, 128 + num2, 128 + num3)
                    except Exception as e:
                        print(e)
                except socket.timeout:
                    print("No data received for TIME_OUT seconds, stopping AGV.")
                    MA.stop()
        except Exception as e:
            print(f"Error handling connection: {e}")
        finally:
            print("Client disconnected, stopping AGV.")
            MA.stop()
            conn.close()


if __name__ == "__main__":
    MA = MyAgv('/dev/ttyACM0', 115200)
    try:
        start_server()
    except KeyboardInterrupt:
        print("Server shutting down.")
    finally:
        MA.stop()
        print("AGV stopped.")
