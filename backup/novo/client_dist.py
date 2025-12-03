import socket
import numpy as np
import pickle
import sys
import threading
import matplotlib.pyplot as plt
import seaborn as sns
import math
import time

# --- Configurações dos Workers ---
# Lista de todos os servidores/workers disponíveis para a distribuição da carga de trabalho
WORKERS = [
    ('127.0.0.1', 65431),  # Worker 1
    ('127.0.0.1', 65432)   # Worker 2
]

# Objeto de resultado thread-safe para armazenar as fatias de C recebidas na ordem correta
class ResultContainer:
    def __init__(self, num_slices):
        self.results = [None] * num_slices # Lista para manter a ordem dos resultados
        self.lock = threading.Lock()
        self.counter = 0

def salvar_txt(nome_arquivo, A, B, C):
    """
    Função para salvar as matrizes A, B e C em um arquivo de texto.
    """
    try:
        with open(nome_arquivo, 'w') as f:
            f.write("=== RELATÓRIO DE MULTIPLICAÇÃO DISTRIBUÍDA ===\n")
            f.write(f"Data/Hora: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"--- MATRIZ A ({A.shape}) ---\n")
            np.savetxt(f, A, fmt='%d', delimiter='\t')
            
            f.write(f"\n--- MATRIZ B ({B.shape}) ---\n")
            np.savetxt(f, B, fmt='%d', delimiter='\t')
            
            f.write(f"\n--- MATRIZ RESULTANTE C ({C.shape}) ---\n")
            np.savetxt(f, C, fmt='%d', delimiter='\t')
            
        print(f"\n[ARQUIVO] Relatório salvo com sucesso em: {nome_arquivo}")
    except Exception as e:
        print(f"[ERRO] Falha ao salvar arquivo TXT: {e}")

def plot_matrix(matrix, title):
    """
    Função para plotar a matriz C e seus elementos.
    """
    plt.figure(figsize=(10, 8))
    # O heatmap é ideal para visualizar a distribuição dos valores em uma matriz
    sns.heatmap(matrix, cmap="viridis", annot=False, cbar=True)
    plt.title(title, fontsize=16)
    plt.xlabel(f"Colunas ({matrix.shape[1]})")
    plt.ylabel(f"Linhas ({matrix.shape[0]})")
    plt.show()

def send_and_receive_slice(worker_addr, a_slice, matrix_b, result_container, slice_index):
    """
    Função executada por cada thread, lidando com um Worker específico.
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
        
        # Envia o tamanho dos dados + dados
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
            print("\n" + "="*50)
            print("--- INSTRUÇÕES ---")
            print(f"Servidores Workers disponíveis: {len(WORKERS)} (Portas: {', '.join(str(p) for _, p in WORKERS)})")
            print("Digite os tamanhos das matrizes A e B no formato 'RxC R2xC2'.")
            print("Exemplo para log solicitado: 100x50 50x40")
            print("Digite 'sair' para encerrar o programa.")
            print("-" * 20)

            message = input(">> Tamanhos Matrizes A e B: ").strip()
            
            if message.lower() == 'sair':
                print("Encerrando o Cliente. Até logo!")
                break
                
            if not message:
                continue

            # 1. Gerar Matrizes A e B no Cliente
            try:
                parts = message.split()
                (rows_a, cols_a), (rows_b, cols_b) = [tuple(map(int, p.split('x'))) for p in parts]
            except ValueError:
                print("Formato de entrada inválido. Use 'RxC R2xC2' (ex: 10x10 10x10).")
                continue

            if cols_a != rows_b:
                print(f"ERRO: A multiplicação não é possível. Colunas de A ({cols_a}) devem ser iguais às linhas de B ({rows_b}).")
                continue
                
            num_workers = len(WORKERS)
            if rows_a < num_workers:
                 print(f"AVISO: Linhas de A ({rows_a}) < Workers ({num_workers}). Usando {rows_a} workers.")
                 num_workers = rows_a
            
            print(f"[CLIENTE] Gerando A ({rows_a}x{cols_a}) e B ({rows_b}x{cols_b})...")
            # Instanciação com np.random.randint
            matrix_a = np.random.randint(0, 100, size=(rows_a, cols_a), dtype=np.int32)
            matrix_b = np.random.randint(0, 100, size=(rows_b, cols_b), dtype=np.int32)
            
            # 2. Divisão da Carga de Trabalho
            slices_a = np.array_split(matrix_a, num_workers, axis=0)
            
            result_container = ResultContainer(len(slices_a))
            worker_threads = []
            
            print(f"[CLIENTE] Distribuindo {len(slices_a)} fatias para Workers...")
            
            # 3. Execução de Multiplicação de Matrizes (Paralela)
            t_start = time.time()
            for i, a_slice in enumerate(slices_a):
                worker_index = i % len(WORKERS)
                worker_addr = WORKERS[worker_index]
                
                t = threading.Thread(
                    target=send_and_receive_slice, 
                    args=(worker_addr, a_slice, matrix_b, result_container, i)
                )
                worker_threads.append(t)
                t.start()
            
            # Aguarda a conclusão de todas as threads
            for t in worker_threads:
                t.join()
            
            t_end = time.time()

            # Verifica se todos os resultados foram recebidos
            if result_container.counter != len(slices_a):
                 print("\nERRO FATAL: Falha ao receber todos os resultados parciais.")
                 continue

            # 4. Concatenar resultados parciais (REQUISITO: Concatenação no Cliente)
            matrix_c = np.concatenate(result_container.results, axis=0)
            
            # Apresenta o Resultado e os Logs (REQUISITO: Log no Terminal)
            print("\n--- LOGS DA MULTIPLICAÇÃO DISTRIBUÍDA ---")
            print(f"Tempo Total: {t_end - t_start:.4f}s")
            print(f"Matriz A: {matrix_a.shape}")
            print(f"Matriz B: {matrix_b.shape}")
            print(f"Matriz Resultado C: {matrix_c.shape}")
            print("------------------------------------------")
            
            # --- NOVO: GERAR ARQUIVO TXT ---
            nome_arquivo_txt = f"resultado_matriz_{rows_a}x{cols_b}.txt"
            salvar_txt(nome_arquivo_txt, matrix_a, matrix_b, matrix_c)

            # Plota a matriz C (REQUISITO: Plotar Imagem)
            # Nota: O plot bloqueia a execução até a janela ser fechada
            print("[VISUALIZAÇÃO] Gerando gráfico heatmap...")
            plot_matrix(matrix_c, f"Matriz Resultado C ({matrix_c.shape}) - Calculada por {num_workers} Workers")
            
            print("\nProcessamento concluído.")

        except ConnectionRefusedError:
            print(f"\nERRO: Conexão recusada. Verifique se todos os Worker Servers ({WORKERS[0][1]} e {WORKERS[1][1]}) estão ativos.")
        except Exception as e:
            print(f"\nOcorreu um erro no cliente: {e}")
            break

if __name__ == '__main__':
    client_program()