import socket
import numpy as np
import time
import multiprocessing as mp
from multiprocessing.dummy import Pool as ThreadPool

HOST = "127.0.0.1"
PORT = 5000


# --------------------------
# 1) Multiplicação SERIAL
# --------------------------
def mult_serial(A, B):
    return A @ B


# --------------------------
# 2) Multiplicação THREADS
# --------------------------
def worker_thread(args):
    A_row, B = args
    return A_row @ B

def mult_threads(A, B, num_workers=4):
    pool = ThreadPool(num_workers)
    result = pool.map(worker_thread, [(A[i], B) for i in range(A.shape[0])])
    pool.close()
    pool.join()
    return np.array(result)


# --------------------------
# 3) Multiplicação PROCESSOS
# --------------------------
def worker_process(args):
    A_chunk, B = args
    return A_chunk @ B

def mult_parallel(A, B, num_workers=4):
    pool = mp.Pool(num_workers)
    chunk_size = len(A) // num_workers
    chunks = [A[i:i+chunk_size] for i in range(0, len(A), chunk_size)]
    results = pool.map(worker_process, [(chunk, B) for chunk in chunks])
    pool.close()
    pool.join()
    return np.vstack(results)


# --------------------------
# SERVIDOR PRINCIPAL
# --------------------------
def handle_client(conn):
    data = conn.recv(1024)
    if not data:
        conn.close()
        return

    size = int(data.decode().strip())

    # gerar matrizes
    A = np.random.randint(0, 10, (size, size))
    B = np.random.randint(0, 10, (size, size))

    # --- Serial ---
    t0 = time.time()
    mult_serial(A, B)
    t_serial = time.time() - t0

    # --- Threads ---
    t0 = time.time()
    mult_threads(A, B)
    t_threads = time.time() - t0

    # --- Paralelo (Processos) ---
    t0 = time.time()
    mult_parallel(A, B)
    t_parallel = time.time() - t0

    # Speedups
    s_threads = t_serial / t_threads if t_threads > 0 else 0
    s_parallel = t_serial / t_parallel if t_parallel > 0 else 0

    # Montar tabela final
    tabela = (
        "===== RESULTADO =====\n"
        f"SERIAL     Tempo = {t_serial:.6f}s   Speedup = 1.0000\n"
        f"THREADS    Tempo = {t_threads:.6f}s   Speedup = {s_threads:.4f}\n"
        f"PARALLEL   Tempo = {t_parallel:.6f}s   Speedup = {s_parallel:.4f}\n"
        "=====================\n"
    )

    conn.sendall(tabela.encode())
    conn.close()


def start_server():
    print(f"Servidor rodando em {HOST}:{PORT}")
    
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    server.bind((HOST, PORT))
    server.listen()

    while True:
        conn, addr = server.accept()
        handle_client(conn)


if __name__ == "__main__":
    mp.freeze_support()   # Necessário no Windows
    start_server()
