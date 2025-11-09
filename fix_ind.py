import os

def substituir_texto_em_arquivos(caminho_da_pasta):
    try:
        # Verifica se o caminho da pasta existe
        if not os.path.isdir(caminho_da_pasta):
            print(f"Erro: A pasta '{caminho_da_pasta}' não foi encontrada.")
            return

        # Lista todos os arquivos no diretório
        for nome_do_arquivo in os.listdir(caminho_da_pasta):
            # Verifica se o arquivo tem a extensão .txt
            if nome_do_arquivo.endswith(".txt"):
                caminho_do_arquivo = os.path.join(caminho_da_pasta, nome_do_arquivo)
                
                try:
                    # Abre o arquivo para leitura
                    with open(caminho_do_arquivo, 'r', encoding='utf-8') as arquivo:
                        conteudo = arquivo.read()

                    # Realiza as substituições no conteúdo
                    novo_conteudo = conteudo.replace('15 ', '0 ').replace('16 ', '1 ')

                    # Se o conteúdo foi alterado, salva o arquivo
                    if novo_conteudo != conteudo:
                        # Abre o mesmo arquivo para escrita (sobrescrevendo o original)
                        with open(caminho_do_arquivo, 'w', encoding='utf-8') as arquivo:
                            arquivo.write(novo_conteudo)
                        print(f"Texto substituído em: {nome_do_arquivo}")
                    else:
                        print(f"Nenhum texto para substituir em: {nome_do_arquivo}")

                except Exception as e:
                    print(f"Não foi possível processar o arquivo {nome_do_arquivo}. Erro: {e}")

        print("\nProcesso concluído!")

    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")


# --- COMO USAR O SCRIPT ---
# 1. Copie e cole este script em um arquivo .py (por exemplo, 'substituir.py').
# 2. Altere o valor da variável 'pasta_alvo' para o caminho da sua pasta.
#    - No Windows, o caminho pode ser algo como: 'C:\\Users\\SeuUsuario\\Documentos\\MeusTextos'
#    - No macOS ou Linux, algo como: '/home/seu_usuario/documentos/meus_textos'
# 3. Execute o script.

if __name__ == "__main__":
    # IMPORTANTE: Substitua pelo caminho da sua pasta
    pasta_alvo = r'C:\Users\GABRIEL\Downloads\LRI_SW_FINAL_PRJCT\datasets\final_dataset\train\labels' 
    
    substituir_texto_em_arquivos(pasta_alvo)