import socket
import pickle
import struct
import threading
import time
import io
import numpy as np
import matplotlib
import os
# Backend 'Agg' para gerar gráficos sem GUI
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing

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
# TAREFAS (Serial, Thread, Processo)
# =============================================================================

def perform_matrix_op(m_size):
    """
    Gera matrizes e calcula.
    Retorna apenas o shape para evitar overhead de I/O desnecessário.
    """
    # Recria seed para garantir aleatoriedade em threads/processos distintos
    np.random.seed() 
    mat_a = np.random.rand(m_size, m_size)
    mat_b = np.random.rand(m_size, m_size)
    
    # Realiza a multiplicação (CPU-bound)
    res = np.dot(mat_a, mat_b)
    
    return res.shape

# Wrapper para o ProcessPool (deve ser top-level para pickle funcionar)
def task_process_heavy(m_size):
    return perform_matrix_op(m_size)

# =============================================================================
# GERADORES DE GRÁFICOS E TABELAS
# =============================================================================

def generate_table_str(t_serial, t_thread, t_process):
    """Gera a tabela formatada exatamente como solicitado."""
    s_serial = 1.0
    # Evita divisão por zero ou números infinitos
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
    """Gráfico de barras para uma única execução (Opção 1)."""
    labels = ['SERIAL', 'CONCORRÊNCIA', 'PARALELISMO']
    colors = ['#3498db', '#f1c40f', '#e74c3c'] # Azul, Amarelo, Vermelho
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(labels, times, color=colors)
    
    ax.set_ylabel('Tempo Médio (s)')
    ax.set_title(f'Desempenho: Matriz {m_size}x{m_size}')
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Adiciona valores no topo das barras
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}s',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    data = buf.read()
    plt.close(fig)
    return data

def generate_consolidated_plot(history):
    """Gráfico de barras clusterizado (Opção 2)."""
    if not history:
        return None

    # Ordena histórico por tamanho
    history.sort(key=lambda x: x[0])
    
    sizes = [str(h[0]) for h in history] # Labels eixo X
    t_serial = [h[1] for h in history]
    t_thread = [h[2] for h in history]
    t_process = [h[3] for h in history]

    x = np.arange(len(sizes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width, t_serial, width, label='SERIAL', color='#3498db')
    rects2 = ax.bar(x, t_thread, width, label='CONCORRÊNCIA', color='#f1c40f')
    rects3 = ax.bar(x + width, t_process, width, label='PARALELISMO', color='#e74c3c')

    ax.set_xlabel('Tamanho da Matriz (NxN)')
    ax.set_ylabel('Tempo (s)')
    ax.set_title('Análise Consolidada (Overhead visível em N pequeno)')
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
# LÓGICA DO SERVIDOR
# =============================================================================

HOST = '0.0.0.0'
PORT = 5000

HISTORY = []
HISTORY_LOCK = threading.Lock()

def warm_up_system(cpu_pool, thread_pool):
    """
    Executa cálculos descartáveis para carregar bibliotecas,
    iniciar processos e 'aquecer' o sistema.
    """
    print("[INIT] Aquecendo motores (Warm-up)...")
    dummy_size = 100
    
    # Serial Warmup
    perform_matrix_op(dummy_size)
    
    # Thread Warmup
    thread_pool.submit(perform_matrix_op, dummy_size).result()
    
    # Process Warmup
    cpu_pool.submit(task_process_heavy, dummy_size).result()
    print("[INIT] Sistema aquecido e pronto.")

def measure_execution(executor, func, m_size, mode='serial'):
    """
    Executa a função múltiplas vezes para N pequeno e tira a média.
    Para N grande, executa apenas uma vez.
    Isso suaviza o jitter do SO.
    """
    # Se a matriz é pequena (< 300), rodamos mais vezes para ter precisão
    # Se é grande, 1 vez basta (pois demora segundos)
    runs = 5 if m_size < 300 else 1
    total_time = 0.0

    for _ in range(runs):
        t0 = time.time()
        
        if mode == 'serial':
            func(m_size)
        else:
            # executor é thread_pool ou cpu_pool
            future = executor.submit(func, m_size)
            future.result() # Bloqueia para garantir que terminou
            
        total_time += (time.time() - t0)

    return total_time / runs

def handle_client(conn, addr, cpu_pool, thread_pool):
    print(f"[CONEXÃO] {addr} conectado.")
    try:
        while True:
            raw_data = recv_msg(conn)
            if not raw_data: break
            
            req = pickle.loads(raw_data)
            response = {}

            if req['acao'] == '1':
                m_size = req['valor']
                print(f"[*] Matriz {m_size}: Calculando...")
                
                # --- MEDIÇÃO COM MÉDIA E SUAVIZAÇÃO ---
                
                # 1. Serial
                t_serial = measure_execution(None, perform_matrix_op, m_size, mode='serial')
                
                # 2. Concorrência
                t_thread = measure_execution(thread_pool, perform_matrix_op, m_size, mode='pool')
                
                # 3. Paralelismo
                t_process = measure_execution(cpu_pool, task_process_heavy, m_size, mode='pool')

                print(f"    -> Resultados (Média): S={t_serial:.5f}s | T={t_thread:.5f}s | P={t_process:.5f}s")

                # Salvar Histórico
                with HISTORY_LOCK:
                    HISTORY.append((m_size, t_serial, t_thread, t_process))

                # Gerar Respostas
                txt_tabela = generate_table_str(t_serial, t_thread, t_process)
                img_bytes = generate_single_plot(m_size, [t_serial, t_thread, t_process])

                response = {'tabela': txt_tabela, 'imagem': img_bytes}

            elif req['acao'] == '2':
                # Consolidado
                with HISTORY_LOCK:
                    data_snapshot = list(HISTORY)
                
                img_bytes = generate_consolidated_plot(data_snapshot)
                
                txt_head = "===== HISTÓRICO CONSOLIDADO =====\n"
                txt_head += f"{'Size':<8} {'Serial(s)':<12} {'Conc(s)':<12} {'Paral(s)':<12}\n"
                txt_body = ""
                for item in data_snapshot:
                    txt_body += f"{item[0]:<8} {item[1]:.4f}       {item[2]:.4f}       {item[3]:.4f}\n"
                txt_body += "================================="
                
                response = {'tabela': txt_head + txt_body, 'imagem': img_bytes}

            send_msg(conn, response)

    except Exception as e:
        print(f"[ERRO] {addr}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()

def main():
    max_workers = multiprocessing.cpu_count()
    print(f"[INIT] Servidor com {max_workers} núcleos.")

    with ProcessPoolExecutor(max_workers=max_workers) as proc_executor:
        with ThreadPoolExecutor(max_workers=max_workers) as thread_executor:
            
            # --- FASE DE AQUECIMENTO ---
            warm_up_system(proc_executor, thread_executor)
            
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
                        args=(conn, addr, proc_executor, thread_executor)
                    )
                    t.daemon = True
                    t.start()
            except KeyboardInterrupt:
                print("\n[SHUTDOWN] Parando servidor...")

if __name__ == "__main__":
    main()