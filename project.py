# CRISP-DM: preparação de dados fase 3
import pandas as pd
import numpy as np
import pickle

# CRISP-DM: análise de dados fase 3 - 123
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


NOMEARQUIVO = "AdClick.csv"
NOMEMODELO = "AdClick.pickle"

def carregarDados(nomeArquivo):
    
    dados = None
    try:
        dados = pd.read_csv(nomeArquivo, sep=",")

    except:
        print("Não foi possivel carregar os dados")
    finally:
        print("")

    return dados

def prepararDados(dados):
    
    # retorna dados básicos sobre o arquivo/dataset carregado
    # qtd colunas, qtd registros não nulos, tipo das colunas
    print(dados.info())

    # 5 primeiros registros do dataset
    print(dados.head())

    # 5 ultimos registros do dataset
    print(dados.tail())

    # retorna estatistica básicas dos campos do dadaset
    print(dados.describe())

    dados['TimeSpentSeconds'] = dados['TimeSpentSeconds'].round(2)

    return dados


def visualizarDados(dados):

    dados['Clicked'].hist()
    plt.show()

    dados['AdType'].hist()
    plt.show()

    dados[["TimeSpentSeconds", "PagesVisited"]].plot()
    plt.show()

    target_counts = dados['Clicked'].value_counts()
    # Graf. de Pizza / TARGET
    plt.figure(figsize=(6,6))
    plt.pie(target_counts, labels=target_counts.index, autopct='%1.1f%%', startangle=90, colors=['red','green'])
    plt.title('Distribuição da Variável "Target"')
    plt.show()


def separarDados(dados):
    print("Separar os dados de treino e teste")

    # separando o objetivo de estratificação [0,1] -- lista 70% e 30%
    Y = dados['Clicked'] # variavel dependente/alvo/target
    X = dados[['TimeSpentSeconds','PagesVisited']] # variaveis independentes/feature
    print(X.head())
    #dadosTreino, dadosTeste = train_test_split(dados, test_size=0.3, train_size=0.7, stratify=Y)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, 
                                                        test_size=0.3,
                                                        train_size=0.7, 
                                                        shuffle=True, 
                                                        random_state=42, 
                                                        stratify=Y)

    return x_train, x_test, y_train, y_test

def treinarModelo(x_train, x_test, y_train, y_test, nomeModelo=NOMEMODELO):
    print("\n--- Fase 4: Modelagem (Logistic Regression) ---")
    modelo = LogisticRegression(max_iter=1000)
    modelo.fit(x_train, y_train)

    print("\n--- Fase 5: Avaliação do Modelo ---")
    y_pred = modelo.predict(x_test)

    acuracia = accuracy_score(y_test, y_pred)
    print(f"Acurácia: {acuracia:.4f}")

    print("\nMatriz de Confusão:")
    print(confusion_matrix(y_test, y_pred))

    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))

    # Salva o modelo treinado
    with open(nomeModelo, 'wb') as arquivo:
        pickle.dump(modelo, arquivo)
    print(f"\nModelo salvo com sucesso em: {nomeModelo}")

    return modelo

# carregar o conjunto de dados para tretamento - fase 2 do CRISP-DM
dados = carregarDados(NOMEARQUIVO)

if dados is not None:

    # tratamento de dados - fase 3 do CRISP-DM
    dados = prepararDados(dados)

    # visualizar alguns dados de modo gráfico
    visualizarDados(dados)

    # separar os dados de treino e teste
    x_train, x_test, y_train, y_test = separarDados(dados)

    # treinar e avaliar modelo Logistic Regression - fases 4 e 5 do CRISP-DM
    modelo = treinarModelo(x_train, x_test, y_train, y_test)
