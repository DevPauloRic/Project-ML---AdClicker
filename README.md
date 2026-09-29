# Project ML - AdClicker

Este projeto tem como objetivo prever se um usuário irá clicar ou não em um anúncio (`Clicked`: 0 ou 1) com base no comportamento de navegação no site (como tempo gasto na página e quantidade de páginas visitadas), utilizando técnicas de Machine Learning e a metodologia **CRISP-DM**.

---

## 📌 Metodologia (CRISP-DM)

O desenvolvimento do fluxo de dados e modelo foi estruturado de acordo com as etapas do CRISP-DM:

1. **Entendimento do Negócio:** Compreender o comportamento de conversão/clique de usuários em anúncios publicitários.
2. **Entendimento dos Dados (Fase 2):** Carga do dataset `AdClick.csv` e análise exploratória inicial (`info()`, `describe()`, distribuição das classes).
3. **Preparação dos Dados (Fase 3):** Tratamento de dados numéricos (arredondamento de `TimeSpentSeconds`), seleção de variáveis preditoras (`X`) e alvo (`Y`), visualização gráfica e divisão estratificada dos dados em treino (70%) e teste (30%).
4. **Modelagem (Fase 4):** Treinamento do algoritmo de classificação supervisionada **Regressão Logística** (`LogisticRegression`).
5. **Avaliação (Fase 5):** Análise de métricas de desempenho no conjunto de teste, incluindo Acurácia, Matriz de Confusão e Relatório de Classificação (Precision, Recall, F1-Score).
6. **Implantação (Fase 6):** Exportação e persistência do modelo treinado em formato binário (`AdClick.pickle`) para uso futuro em predições.

---

## 📂 Estrutura do Projeto

```text
├── AdClick.csv          # Conjunto de dados contendo o histórico de acessos e cliques
├── AdClick.pickle       # Modelo de Machine Learning treinado e serializado
├── project.py           # Script principal com todo o pipeline (carga, análise, treino e avaliação)
├── requirements.txt     # Dependências e bibliotecas necessárias para executar o projeto
├── .gitignore           # Arquivos e diretórios ignorados pelo Git (ex: .venv/)
└── README.md            # Documentação do projeto
```

---

## 🛠️ Tecnologias Utilizadas

- **Python 3**
- **Pandas:** Manipulação e análise estruturada de dados
- **NumPy:** Suporte a operações numéricas e arrays
- **Matplotlib:** Geração de gráficos para análise exploratória
- **Scikit-Learn:** Divisão de treino/teste, modelo de Regressão Logística e cálculo de métricas
- **Pickle:** Serialização e salvamento do modelo de Machine Learning

---

## 🚀 Como Executar o Projeto

### 1. Clonar o repositório
```bash
git clone https://github.com/DevPauloRic/Project-ML---AdClicker.git
cd Project-ML---AdClicker
```

### 2. Criar e ativar o ambiente virtual (Recomendado)
- **No Windows:**
  ```bash
  python -m venv .venv
  .venv\Scripts\activate
  ```
- **No Linux/macOS:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### 3. Instalar as dependências
```bash
pip install -r requirements.txt
```

### 4. Executar o script principal
```bash
python project.py
```

O script irá:
- Exibir os resumos estatísticos no terminal;
- Gerar os gráficos de distribuição das variáveis e do público-alvo;
- Realizar o treinamento e imprimir as métricas de avaliação;
- Salvar/atualizar o arquivo do modelo `AdClick.pickle`.
