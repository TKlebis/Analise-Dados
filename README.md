# 📊 Ciência de Dados
![Python Versions](https://img.shields.io/pypi/pyversions/st_pages.svg)
![License](https://img.shields.io/github/license/blackary/st_pages)
![Streamlit versions](https://img.shields.io/badge/streamlit-1.15.0--1.18.0-white.svg)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://tk-dados.streamlit.app/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://tk-dadoss.streamlit.app/)

*Autor:* [Thiago Klebis](https://www.linkedin.com/in/thiagoklebis/)

>O código disponibilizado é uma aplicação web desenvolvida com a biblioteca Streamlit em Python. A aplicação tem o objetivo de realizar análise de dados a partir de um arquivo carregado pelo usuário.

## ⬇️ FUNÇÕES
- **Escolha de sessão:** O usuario deve escolher entre a opção "[Analise](https://tk-dados.streamlit.app/)" e "Treinamento". 
- **Carregar arquivo de dados:** O usuário pode selecionar um arquivo nos formatos CSV, XLSX, JSON ou TXT para carregar na aplicação.
- **Exibir dados do arquivo:** Após o arquivo ser carregado, os primeiros registros são exibidos ao usuário. O número de linhas a serem exibidas pode ser configurado através de um controle deslizante.
- **Excluir/Alterar valor de linha:** O usuário pode interagir com o DataFrame exibido e realizar a exclusão ou alteração de valores de linhas específicas.
- **Verificar Tipos das Variáveis:** O código exibe os tipos das variáveis presentes no DataFrame. O usuário tem a opção de modificar o tipo das variáveis selecionadas.
- **Análise de Variáveis Duplicadas:** O código verifica se existem variáveis duplicadas no DataFrame e, se houver, oferece opções de tratamento, como exclusão das variáveis duplicadas ou manutenção de apenas uma ocorrência.
- **Análise de Dados Missing:** O código realiza uma análise de dados missing, exibindo a quantidade de valores faltantes por variável. O usuário pode selecionar opções de tratamento para variáveis numéricas e categóricas, como exclusão da variável, substituição pela média (para numéricas) ou moda (para categóricas), ou exclusão de linhas.
- **Detecção e Visualização de Outliers:** O código detecta outliers nas variáveis numéricas do DataFrame e os exibe ao usuário. Além disso, é possível selecionar uma variável e aplicar tratamentos, como remoção dos outliers ou substituição por limites.
- **Gerar Gráfico de Barras:** O usuário pode gerar um gráfico de barras a partir dos dados do DataFrame. O gráfico é exibido utilizando a biblioteca Matplotlib.

  ![Design sem nome](https://github.com/TKlebis/Analise-Dados/assets/130613291/a8f77904-3b53-4946-b608-62247d0de48c)


- ## ⬇️ FUNÇÕES
- **Escolha de sessão:** O usuario deve escolher entre a opção "Analise" e "[Treinamento](https://tk-dadoss.streamlit.app/)". 
**Carregar arquivo de dados:** O usuário pode selecionar um arquivo nos formatos CSV, XLSX, JSON ou TXT para carregar na aplicação.
- **Exibir dados do arquivo:** Após o arquivo ser carregado, os primeiros registros são exibidos ao usuário. O número de linhas a serem exibidas pode ser configurado através de um controle deslizante.
- **Excluir/Alterar valor de linha:** O usuário pode interagir com o DataFrame exibido e realizar a exclusão ou alteração de valores de linhas específicas.
- **Verificar Tipos das Variáveis:** O código exibe os tipos das variáveis presentes no DataFrame. O usuário tem a opção de modificar o tipo das variáveis selecionadas.
- **Análise de Variáveis Duplicadas:** O código verifica se existem variáveis duplicadas no DataFrame e, se houver, oferece opções de tratamento, como exclusão das variáveis duplicadas ou manutenção de apenas uma ocorrência.
- **Análise de Dados Missing:** O código realiza uma análise de dados missing, exibindo a quantidade de valores faltantes por variável. O usuário pode selecionar opções de tratamento para variáveis numéricas e categóricas, como exclusão da variável, substituição pela média (para numéricas) ou moda (para categóricas), ou exclusão de linhas.
- **Detecção e Visualização de Outliers:** O código detecta outliers nas variáveis numéricas do DataFrame e os exibe ao usuário. Além disso, é possível selecionar uma variável e aplicar tratamentos, como remoção dos outliers ou substituição por limites.
- **Análise Descritiva:** O aplicativo exibe uma análise descritiva do DataFrame carregado, mostrando estatísticas básicas como média, desvio padrão, mínimo, máximo e quartis.
- **Treinamento de Modelos de Classificação:** O aplicativo permite treinar modelos de classificação usando a biblioteca PyCaret. É possível selecionar as variáveis de entrada, a variável alvo e comparar diferentes modelos de classificação.
- **Inserir novos Dados para previsão:** O aplicativo permite criar um novo Dataframe para que faça uma previsão da variável preditora.
- **Previsão:** Será realizado a previsão com o melhor modelo escolhido após treinamento, será fornecido um Dataframe para avaliar o resultado da variável preditora.
- **Download:** Ao final, tem a possibilidade de baixar o arquivo do resultado da revisão.

  ![Design sem nome (1)](https://github.com/TKlebis/Analise-Dados/assets/130613291/710d42b9-7e8a-4491-a3c0-aa07c7125ea3)

  



