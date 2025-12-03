import socket
import numpy as np
import pickle
import sys
import threading

# --- Configurações dos Workers ---
# Lista de todos os servidores disponíveis para a distribuição da carga de trabalho
WORKERS = [
    ('127.0.0.1', 65431),  # Worker 1
    ('127.0.0.1', 65432)   # Worker 2
]

# Objeto de resultado para armazenar as fatias de C recebidas
# É compartilhado entre as threads
class ResultContainer:
    def __init__(self, num_slices):
        self.results = [None] * num_slices # Usa lista para manter a ordem
        self.lock = threading.Lock()
        self.counter = 0

def send_and_receive_slice(worker_addr, a_slice, matrix_b, result_container, slice_index):
    """
    Função executada por cada thread, lidando com um Worker específico.
    REQUISITO: Estabelecer conexão com Servidores, Enviar, Receber (parcial).
    """
    worker_host, worker_port = worker_addr
    worker_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    try:
        # Estabelece a conexão com o Worker
        worker_socket.connect((worker_host, worker_port))
        
        # Envia a submatriz A e a Matriz B
        data_to_send = pickle.dumps({
            'A_slice': a_slice,
            'B': matrix_b
        })
        
        worker_socket.sendall(str(len(data_to_send)).encode('utf-8') + b'\n')
        worker_socket.sendall(data_to_send)
        
        # Recebe o tamanho do resultado parcial (cabeçalho)
        size_header = b''
        while True:
            chunk = worker_socket.recv(1)
            if not chunk or chunk == b'\n':
                break
            size_header += chunk
        
        if not size_header:
            raise RuntimeError(f"Worker {worker_port} fechou a conexão.")
            
        data_size = int(size_header.decode('utf-8'))
        
        # Recebe os dados serializados do resultado parcial
        data_raw = b''
        bytes_recd = 0
        while bytes_recd < data_size:
            chunk = worker_socket.recv(min(data_size - bytes_recd, 4096))
            if not chunk:
                raise RuntimeError("Conexão interrompida antes de receber todos os dados.")
            data_raw += chunk
            bytes_recd += len(chunk)
            
        # Deserializa o resultado parcial (C_slice)
        c_slice = pickle.loads(data_raw)

        # Armazena o resultado na ordem correta
        with result_container.lock:
            result_container.results[slice_index] = c_slice
            result_container.counter += 1
            print(f"  > Recebido resultado da fatia {slice_index + 1}/{len(result_container.results)} (Worker {worker_port}).")

    except Exception as e:
        print(f"[ERRO THREAD] Falha ao comunicar com Worker {worker_port}: {e}")

    finally:
        worker_socket.close()


def client_program():
    print(f"Coordenador (Cliente) de Multiplicação de Matrizes Distribuída")
    
    # Loop de persistência (REQUISITO: permanecer conectado)
    while True:
        try:
            print("\n--- INSTRUÇÕES ---")
            print(f"Servidores Workers disponíveis: {len(WORKERS)} (Portas: {', '.join(str(p) for _, p in WORKERS)})")
            print("Digite os tamanhos das matrizes A e B no formato 'RxC R2xC2'.")
            print("Exemplo: 100x50 50x70 (Requer 2 workers)")
            print("Para SAIR do programa, digite: sair")
            print("------------------")

            message = input("Tamanhos Matrizes A e B -> ").strip()
            
            if message.lower() == 'sair':
                print("Encerrando o Cliente. Até logo!")
                break
                
            if not message:
                continue

            # 1. Gerar Matrizes A e B no Cliente (REQUISITO 1)
            parts = message.split()
            (rows_a, cols_a), (rows_b, cols_b) = [tuple(map(int, p.split('x'))) for p in parts]
            
            if cols_a != rows_b:
                print(f"ERRO: A multiplicação não é possível. Colunas de A ({cols_a}) devem ser iguais às linhas de B ({rows_b}).")
                continue
                
            num_workers = len(WORKERS)
            if rows_a < num_workers:
                 print(f"AVISO: O número de linhas ({rows_a}) é menor que o número de Workers ({num_workers}). Usando {rows_a} workers.")
                 num_workers = rows_a # Reduz o número de workers se houver poucas linhas
            
            print(f"[CLIENTE] Gerando A ({rows_a}x{cols_a}) e B ({rows_b}x{cols_b})...")
            # Instanciação com np.random.randint
            matrix_a = np.random.randint(0, 100, size=(rows_a, cols_a), dtype=np.int32)
            matrix_b = np.random.randint(0, 100, size=(rows_b, cols_b), dtype=np.int32)
            
            # 2. Divisão da Carga de Trabalho (REQUISITO 2)
            # Divide a Matriz A no número de Workers disponíveis
            slices_a = np.array_split(matrix_a, num_workers, axis=0)
            
            result_container = ResultContainer(len(slices_a))
            worker_threads = []
            
            # 3. Execução de Multiplicação de Matrizes (Paralela)
            # Envia cada fatia para um Worker diferente (circularmente)
            for i, a_slice in enumerate(slices_a):
                worker_index = i % len(WORKERS)
                worker_addr = WORKERS[worker_index]
                
                t = threading.Thread(
                    target=send_and_receive_slice, 
                    args=(worker_addr, a_slice, matrix_b, result_container, i)
                )
                worker_threads.append(t)
                t.start()
            
            # Aguarda a conclusão de todas as threads (Multiplicação)
            for t in worker_threads:
                t.join()
                
            # Verifica se todos os resultados foram recebidos
            if result_container.counter != len(slices_a):
                 print("\nERRO FATAL: Falha ao receber todos os resultados parciais.")
                 continue

            # 4. Concatenar resultados parciais (REQUISITO 4 & 5)
            # A concatenação é segura porque os resultados foram armazenados na ordem correta pelo índice
            matrix_c = np.concatenate(result_container.results, axis=0)
            
            # Apresenta o Resultado
            print("\n--- RESULTADO FINAL COMPILADO PELO CLIENTE ---")
            print(f"Matriz A: {matrix_a.shape}")
            print(f"Matriz B: {matrix_b.shape}")
            print(f"Matriz Resultado C: {matrix_c.shape}")
            print(f"Cálculo distribuído entre {num_workers} Workers finalizado.")
            print("------------------------------------------------\n")


        except Exception as e:
            print(f"\nOcorreu um erro no cliente: {e}")
            break

if __name__ == '__main__':
    client_program()