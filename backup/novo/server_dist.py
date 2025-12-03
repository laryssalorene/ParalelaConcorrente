import socket
import numpy as np
import threading
import pickle
import sys

def worker_handler(conn, addr):
    """Recebe A_slice e B, calcula C_slice e envia de volta."""
    print(f"[WORKER {sys.argv[1]}] Conexão recebida de {addr}")
    
    # Loop persistente para atender múltiplas requisições do mesmo cliente
    while True:
        try:
            # 1. Recebe o tamanho da requisição (cabeçalho)
            size_header = b''
            # Lógica para ler o cabeçalho (tamanho dos dados) até o separador '\n'
            while True:
                chunk = conn.recv(1)
                if not chunk or chunk == b'\n':
                    break
                size_header += chunk
                
            if not size_header:
                break
                
            data_size = int(size_header.decode('utf-8'))
            
            # 2. Recebe os dados serializados (A_slice e Matriz B)
            data_raw = b''
            bytes_recd = 0
            while bytes_recd < data_size:
                chunk = conn.recv(min(data_size - bytes_recd, 4096))
                if not chunk:
                    raise RuntimeError("Conexão interrompida.")
                data_raw += chunk
                bytes_recd += len(chunk)
            
            # 3. Deserializa e realiza a multiplicação
            matrices = pickle.loads(data_raw)
            a_slice = matrices['A_slice']
            matrix_b = matrices['B']
            
            # Realiza a multiplicação (REQUISITO: Multiplicação feita pelo Servidor)
            c_slice = a_slice @ matrix_b 
            
            # 4. Envia o resultado parcial de volta
            result_data = pickle.dumps(c_slice)
            
            # Envia o tamanho dos dados primeiro
            conn.sendall(str(len(result_data)).encode('utf-8') + b'\n') 
            
            # Envia os dados serializados
            conn.sendall(result_data)
            print(f"[WORKER {sys.argv[1]}] Resultado parcial enviado ({c_slice.shape}).")
            
        except Exception as e:
            print(f"[WORKER {sys.argv[1]} ERRO] {e}")
            break

    conn.close()

def start_worker_server():
    # A porta é passada como argumento na linha de comando
    if len(sys.argv) != 2:
        print("Uso: python worker_server.py <PORTA>")
        sys.exit(1)
        
    HOST = '127.0.0.1' 
    PORT = int(sys.argv[1])
    
    worker_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    worker_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        worker_socket.bind((HOST, PORT))
        worker_socket.listen(5)
        print(f"Worker Server rodando na porta {PORT}...")
    except Exception as e:
        print(f"Não foi possível iniciar o Worker Server: {e}")
        sys.exit()

    while True:
        try:
            conn, addr = worker_socket.accept()
            client_thread = threading.Thread(target=worker_handler, args=(conn, addr))
            client_thread.start()
        except KeyboardInterrupt:
            print(f"\nWorker Server {PORT} encerrado.")
            worker_socket.close()
            break

if __name__ == "__main__":
    # Comando para iniciar: python server_dist.py 65431
    # Comando para iniciar: python server_dist.py 65432
    start_worker_server()