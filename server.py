import os

# --- CONFIGURAÇÃO DE AMBIENTE (CRÍTICO) ---
# Força o Numpy a usar apenas 1 thread por operação.
# Isso garante que o modo SERIAL seja puramente single-core (lento),
# permitindo que o paralelismo via Processos (Foster) mostre seu valor.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import socket
import pickle
import struct
import threading
import time
import io
import numpy as np
import matplotlib
# Backend 'Agg' para gerar gráficos sem GUI
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing
from multiprocessing import shared_memory

# =============================================================================
# PROTOCOLO
# =============================================================================

def send_msg(sock, data):
    try:
        msg = pickle.dumps(data)
        sock.sendall(struct.pack('>I', len(msg)) + msg)
    except Exception as e:
        print(f"Erro no envio: {e}")

def recv_msg(sock):
    raw_msglen = recvall(sock, 4)
    if not raw_msglen: return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    return recvall(sock, msglen)

def recvall(sock, n):
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet: return None
        data.extend(packet)
    return data

# =============================================================================
# WORKER: A UNIDADE DE CÁLCULO
# =============================================================================

def worker_chunk_calc(shm_names, shape, start_row, end_row, dtype):
    """
    Worker agnóstico: Pode ser executado por uma Thread ou por um Processo.
    1. Conecta à memória compartilhada (Zero-Copy).
    2. Calcula a fatia designada (Chunk).
    """
    # 1. Conectar à memória compartilhada existente
    try:
        existing_shm_a = shared_memory.SharedMemory(name=shm_names['A'])
        existing_shm_b = shared_memory.SharedMemory(name=shm_names['B'])
        
        # 2. Reconstruir arrays numpy a partir do buffer
        mat_a = np.ndarray(shape, dtype=dtype, buffer=existing_shm_a.buf)
        mat_b = np.ndarray(shape, dtype=dtype, buffer=existing_shm_b.buf)
        
        # 3. Cálculo da Fatia (Tarefa Aglomerada)
        _res_chunk = np.dot(mat_a[start_row:end_row, :], mat_b)
        
        # 4. Fechar acesso
        existing_shm_a.close()
        existing_shm_b.close()
    except Exception as e:
        print(f"Erro no worker: {e}")
        return 0
    
    return (end_row - start_row)

# Wrapper para serialização no ProcessPool
def task_wrapper(args):
    return worker_chunk_calc(*args)

class MatrixManager:
    """Gerencia alocação e limpeza de Shared Memory."""
    def __init__(self, size):
        self.size = size
        self.shape = (size, size)
        self.dtype = np.float64
        self.nbytes = int(size * size * 8)
        self.shm_a = None
        self.shm_b = None

    def allocate(self):
        self.shm_a = shared_memory.SharedMemory(create=True, size=self.nbytes)
        self.shm_b = shared_memory.SharedMemory(create=True, size=self.nbytes)
        arr_a = np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm_a.buf)
        arr_b = np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm_b.buf)
        arr_a[:] = np.random.rand(*self.shape)
        arr_b[:] = np.random.rand(*self.shape)
        return {'A': self.shm_a.name, 'B': self.shm_b.name}

    def cleanup(self):
        if self.shm_a:
            self.shm_a.close()
            self.shm_a.unlink()
        if self.shm_b:
            self.shm_b.close()
            self.shm_b.unlink()

# =============================================================================
# MÉTODOS DE CÁLCULO (SERIAL, CONCORRENTE, PARALELO)
# =============================================================================

def algorithm_serial(shm_names, shape, dtype):
    """
    MÉTODO 1: SERIAL
    Executa o cálculo inteiro em um único núcleo (Main Thread).
    Não há particionamento nem overhead de gerenciamento de tarefas.
    """
    return worker_chunk_calc(shm_names, shape, 0, shape[0], dtype)

def _distribute_tasks_foster(executor, m_size, shm_names, num_workers):
    """Lógica interna de distribuição Foster (PCAM) usada por Concorrente e Paralelo."""
    # Aglomeração: Define o tamanho do bloco
    rows_per_chunk = m_size // num_workers
    futures = []
    
    # Mapeamento: Distribui blocos para workers
    for i in range(num_workers):
        start = i * rows_per_chunk
        end = m_size if i == num_workers - 1 else (i + 1) * rows_per_chunk
        args = (shm_names, (m_size, m_size), start, end, np.float64)
        futures.append(executor.submit(task_wrapper, args))
    
    # Sincronização
    for f in futures:
        f.result()

def algorithm_concurrent(thread_pool, m_size, shm_names, num_workers):
    """
    MÉTODO 2: PURAMENTE CONCORRENTE
    Usa Threads. No Python, threads compartilham o GIL.
    Ideal para I/O, mas em CPU-bound sofrem contenção.
    Aqui aplicamos Foster para dividir a carga entre threads.
    """
    _distribute_tasks_foster(thread_pool, m_size, shm_names, num_workers)

def algorithm_parallel(process_pool, m_size, shm_names, num_workers):
    """
    MÉTODO 3: PARALELO (FOSTER + SHARED MEMORY)
    Usa Processos Independentes (ProcessPoolExecutor).
    Cada processo tem seu próprio GIL e roda em um núcleo físico diferente.
    Usa Memória Compartilhada para evitar cópia de dados (Zero-Copy).
    """
    _distribute_tasks_foster(process_pool, m_size, shm_names, num_workers)

# =============================================================================
# GRÁFICOS E TABELAS
# =============================================================================

def generate_table_str(t_serial, t_thread, t_process):
    s_serial = 1.0
    s_thread = t_serial / t_thread if t_thread > 1e-9 else 1.0
    s_process = t_serial / t_process if t_process > 1e-9 else 1.0

    table = (
        "===== RESULTADO =====\n"
        f"{'':<18} {'Tempo(s)':<14} {'SpeedUp':<10}\n"
        f"{'SERIAL':<16} {t_serial:.6f}s      {s_serial:.4f}\n"
        f"{'CONCORRÊNCIA':<16} {t_thread:.6f}s      {s_thread:.4f}\n"
        f"{'PARALELISMO':<16} {t_process:.6f}s      {s_process:.4f}\n"
        "====================="
    )
    return table

def generate_single_plot(m_size, times):
    labels = ['SERIAL', 'CONCORRÊNCIA', 'PARALELISMO']
    colors = ['#3498db', '#f1c40f', '#e74c3c'] 
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(labels, times, color=colors)
    
    ax.set_ylabel('Tempo (s)')
    ax.set_title(f'Desempenho: Matriz {m_size}x{m_size}')
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}s',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    data = buf.read()
    plt.close(fig)
    return data

def generate_consolidated_plot(history):
    if not history: return None
    history.sort(key=lambda x: x[0])
    
    sizes = [str(h[0]) for h in history]
    t_serial = [h[1] for h in history]
    t_thread = [h[2] for h in history]
    t_process = [h[3] for h in history]

    x = np.arange(len(sizes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, t_serial, width, label='SERIAL', color='#3498db')
    ax.bar(x, t_thread, width, label='CONCORRÊNCIA', color='#f1c40f')
    ax.bar(x + width, t_process, width, label='PARALELISMO', color='#e74c3c')

    ax.set_xlabel('Tamanho da Matriz (NxN)')
    ax.set_ylabel('Tempo (s)')
    ax.set_title('Análise Consolidada (Serial vs Concorrente vs Paralelo)')
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    data = buf.read()
    plt.close(fig)
    return data

# =============================================================================
# SERVIDOR
# =============================================================================

HOST = '0.0.0.0'
PORT = 5000

HISTORY = []
HISTORY_LOCK = threading.Lock()

def handle_client(conn, addr, cpu_pool, thread_pool, num_cores):
    print(f"[CONEXÃO] {addr} conectado.")
    try:
        while True:
            raw_data = recv_msg(conn)
            if not raw_data: break
            
            req = pickle.loads(raw_data)
            response = {}

            if req['acao'] == '1':
                m_size = req['valor']
                print(f"[*] Matriz {m_size}: Iniciando Benchmark...")
                
                mgr = MatrixManager(m_size)
                shm_names = {}
                
                try:
                    shm_names = mgr.allocate()
                    print(f"    -> Dados alocados. Executando métodos...")

                    # 1. SERIAL
                    t0 = time.time()
                    algorithm_serial(shm_names, mgr.shape, mgr.dtype)
                    t_serial = time.time() - t0
                    
                    # 2. CONCORRENTE
                    t0 = time.time()
                    algorithm_concurrent(thread_pool, m_size, shm_names, num_cores)
                    t_thread = time.time() - t0
                    
                    # 3. PARALELO (O foco da questão)
                    t0 = time.time()
                    algorithm_parallel(cpu_pool, m_size, shm_names, num_cores)
                    t_process = time.time() - t0

                except Exception as e:
                    print(f"Erro no cálculo: {e}")
                    t_serial, t_thread, t_process = 0, 0, 0
                finally:
                    mgr.cleanup()
                
                print(f"    -> S={t_serial:.4f}s | T={t_thread:.4f}s | P={t_process:.4f}s")
                
                with HISTORY_LOCK:
                    HISTORY.append((m_size, t_serial, t_thread, t_process))

                txt_tabela = generate_table_str(t_serial, t_thread, t_process)
                img_bytes = generate_single_plot(m_size, [t_serial, t_thread, t_process])
                response = {'tabela': txt_tabela, 'imagem': img_bytes}

            elif req['acao'] == '2':
                with HISTORY_LOCK:
                    data = list(HISTORY)
                img = generate_consolidated_plot(data)
                
                txt_head = "===== HISTÓRICO CONSOLIDADO =====\n"
                txt_head += f"{'Size':<8} {'Serial(s)':<10} {'Conc(s)':<10} {'Paral(s)':<10} {'SpeedUp(C)':<12} {'SpeedUp(P)':<12}\n"
                txt_body = ""
                for item in data:
                    size, t_s, t_c, t_p = item
                    # Calcula SpeedUps relativos ao Serial
                    sp_c = t_s / t_c if t_c > 1e-9 else 0.0
                    sp_p = t_s / t_p if t_p > 1e-9 else 0.0
                    
                    txt_body += f"{size:<8} {t_s:<10.4f} {t_c:<10.4f} {t_p:<10.4f} {sp_c:<12.4f} {sp_p:<12.4f}\n"
                txt_body += "================================="
                response = {'tabela': txt_head + txt_body, 'imagem': img}

            send_msg(conn, response)

    except Exception as e:
        print(f"[ERRO] {addr}: {e}")
    finally:
        conn.close()

def main():
    num_cores = multiprocessing.cpu_count()
    print(f"[INIT] Servidor com {num_cores} núcleos.")
    print("[INIT] Métodos carregados: Serial, Concorrente e Paralelo (Foster).")

    with ProcessPoolExecutor(max_workers=num_cores) as proc_executor:
        with ThreadPoolExecutor(max_workers=num_cores) as thread_executor:
            
            # Warm-up
            print("[INIT] Executando Warm-up...")
            try:
                mgr = MatrixManager(100)
                names = mgr.allocate()
                algorithm_parallel(proc_executor, 100, names, num_cores)
                mgr.cleanup()
            except: pass
            
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((HOST, PORT))
            server.listen()
            print(f"[SERVER] Ouvindo em {HOST}:{PORT}")

            try:
                while True:
                    conn, addr = server.accept()
                    t = threading.Thread(
                        target=handle_client, 
                        args=(conn, addr, proc_executor, thread_executor, num_cores)
                    )
                    t.daemon = True
                    t.start()
            except KeyboardInterrupt:
                print("\n[SHUTDOWN] Parando...")

if __name__ == "__main__":
    main()