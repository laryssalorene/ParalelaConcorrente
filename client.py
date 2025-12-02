import socket
import pickle
import struct
import os
import time

# =============================================================================
# PROTOCOLO DE REDE (ALINHADO COM O SERVIDOR)
# =============================================================================

def send_msg(sock, data):
    """Empacota e envia dados: 4 bytes de tamanho + dados pickle."""
    msg = pickle.dumps(data)
    # '>I' = Big-endian Unsigned Int (4 bytes)
    sock.sendall(struct.pack('>I', len(msg)) + msg)

def recv_msg(sock):
    """Recebe mensagem: lê 4 bytes de tamanho, depois lê o corpo."""
    # 1. Ler o cabeçalho (tamanho)
    raw_msglen = recvall(sock, 4)
    if not raw_msglen: return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    
    # 2. Ler o corpo da mensagem (payload)
    return recvall(sock, msglen)

def recvall(sock, n):
    """Garante o recebimento de exatamente n bytes."""
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet: return None
        data.extend(packet)
    return data

# =============================================================================
# LÓGICA DO CLIENTE
# =============================================================================

HOST = "127.0.0.1"
PORT = 5000

def main():
    try:
        # Cria o socket e conecta
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((HOST, PORT))
        print(f"[*] Conectado ao servidor {HOST}:{PORT}")
    except ConnectionRefusedError:
        print("ERRO: Não foi possível conectar. O servidor está rodando?")
        return

    try:
        while True:
            print("\n" + "="*30)
            print("MENU")
            print("1 - Realizar multiplicação")
            print("2 - Resultados consolidados")
            print("sair - Encerrar")
            print("="*30)
            
            opcao = input("Opção: ").strip().lower()

            if opcao == 'sair':
                break

            requisicao = {}

            # ETAPA 1: Preparar o pacote de dados (Dicionário)
            if opcao == '1':
                try:
                    val = input("Digite o tamanho da matriz (M): ")
                    m_size = int(val)
                    requisicao = {'acao': '1', 'valor': m_size}
                except ValueError:
                    print("Erro: Digite um número inteiro.")
                    continue
            
            elif opcao == '2':
                requisicao = {'acao': '2'}
            
            else:
                print("Opção inválida!")
                continue

            # ETAPA 2: Enviar requisição completa
            print(f"[*] Enviando requisição...")
            send_msg(client_socket, requisicao)

            # ETAPA 3: Receber resposta completa
            print("[*] Processando no servidor... aguarde.")
            dados_bytes = recv_msg(client_socket)
            
            if dados_bytes is None:
                print("Servidor encerrou a conexão.")
                break

            resposta = pickle.loads(dados_bytes)

            # ETAPA 4: Exibir Resultados
            if 'tabela' in resposta:
                print("\n" + resposta['tabela'])
            
            if 'imagem' in resposta and resposta['imagem']:
                if opcao == '1':
                    nome_arq = f"grafico_M{requisicao['valor']}.png"
                else:
                    nome_arq = "grafico_consolidado.png"
                
                with open(nome_arq, "wb") as f:
                    f.write(resposta['imagem'])
                print(f"[V] Gráfico salvo como: {nome_arq}")
                
                # Tenta abrir a imagem automaticamente (Opcional - Funciona no Windows)
                try:
                    if os.name == 'nt': os.startfile(nome_arq) 
                except: pass

    except Exception as e:
        print(f"Erro na execução: {e}")
    finally:
        client_socket.close()
        print("Conexão encerrada.")

if __name__ == "__main__":
    main()