import socket

HOST = "127.0.0.1"
PORT = 5000

size = input("Digite o tamanho da matriz: ")

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))

s.sendall(size.encode())

resposta = s.recv(1000000)
print(resposta.decode())

s.close()
