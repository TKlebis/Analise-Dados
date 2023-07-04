import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import re
import time
import io
import base64
import tempfile
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from pycaret.classification import setup, compare_models, pull, finalize_model, predict_model, evaluate_model
from pycaret.classification import *
from streamlit_option_menu import option_menu
from plotly.subplots import make_subplots

st.set_page_config(page_title="Análise de Dados", page_icon="Dice.png")

selected3 = option_menu(None, [ "Análise", 'Treinamento'],
                        icons=["search", 'bi-diagram-3-fill'],
                        menu_icon="cast", default_index=0, orientation="horizontal",
                        styles={
                            "container": {"padding": "0!important", "background-color": "#fafafa"},
                            "icon": {"color": "orange", "font-size": "25px"},
                            "nav-link": {"font-size": "25px", "text-align": "left", "margin": "0px",
                                         "--hover-color": "#eee"},
                            "nav-link-selected": {"background-color": "green"},
                        })


# Carrega o arquivo na sessão
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

if selected3 == "Treinamento":
    uploaded_file = st.file_uploader("Carregar arquivo de dados", type=["csv", "xlsx", "json", "txt"])


    if uploaded_file is not None:
        file_extension = uploaded_file.name.split(".")[-1]

        if file_extension == "csv":
             df = pd.read_csv(uploaded_file, decimal=',')
        elif file_extension == "xlsx":
            df = pd.read_excel(uploaded_file)
        elif file_extension == "json":
            df = pd.read_json(uploaded_file)
        elif file_extension == "txt":
            df = pd.read_table(uploaded_file)

        st.session_state.uploaded_file = df
        st.success("Arquivo carregado com sucesso!")
        num_rows = st.slider("Número de linhas a serem exibidas", min_value=1, max_value=len(df), value=10)
        st.write(df.head(num_rows))
        
        
        # Exibe o número de linhas e colunas
        num_rows, num_cols = df.shape
        col1,col2= st.columns(2)
        col1.metric("Número de Linhas", f"{num_rows}")
        col2.metric("Número de Colunas", f"{num_cols}")

        # Exibir o DataFrame interativo
        if st.checkbox("🧯 Excluir/Alterar valor de linha"):
            st.subheader("Visualizar e interagir com o DataFrame")

            # Verificar se o DataFrame modificado já foi armazenado
            if 'modified_df' not in st.session_state:
                st.session_state.modified_df = df.copy()

            # Selecionar ação a ser executada
            action = st.radio("Selecione a ação", ["Excluir Dados", "Alterar Dados"])

            if action == "Excluir Dados":
                # Selecionar linha a ser excluída
                row_to_delete = st.number_input("Selecione o índice da linha a ser excluída", min_value=0, max_value=len(st.session_state.modified_df)-1, value=0, step=1)

                if st.button("Aplicar"):
                    # Excluir a linha selecionada
                    st.session_state.modified_df = st.session_state.modified_df.drop(row_to_delete)
                    st.success("Linha excluída com sucesso.")
                    st.write(st.session_state.modified_df)

            elif action == "Alterar Dados":
                # Selecionar linha para alterar dados
                row_to_modify = st.number_input("Selecione o índice da linha para alterar dados", min_value=0, max_value=len(st.session_state.modified_df)-1, value=0, step=1)
                # Selecionar coluna para alterar dados
                column_to_modify = st.selectbox("Selecione a coluna para alterar dados", st.session_state.modified_df.columns)

                # Valor atual da célula selecionada
                current_value = st.session_state.modified_df.loc[row_to_modify, column_to_modify]

                # Novo valor a ser atribuído
                new_value = st.text_input("Digite o novo valor", current_value)

                if st.button("Alterar"):
                    # Alterar o valor na célula selecionada
                    st.session_state.modified_df.loc[row_to_modify, column_to_modify] = new_value
                    st.success("Valor alterado com sucesso.")
                    st.write(st.session_state.modified_df)

        # Verificar tipos das variáveis
        if st.checkbox("📋 Verificar Tipos das Variáveis"):
            st.subheader("Tipos das Variáveis")
            st.write(df.dtypes)

            # Modificar tipo das variáveis
            if st.checkbox("Modificar Tipo das Variáveis"):
                st.subheader("Modificar Tipo das Variáveis")

                # Verificar se o DataFrame foi modificado interativamente
                if 'modified_df' in st.session_state:
                    df = st.session_state.modified_df

                # Selecionar variáveis para modificar o tipo
                variables_to_modify = st.multiselect("Selecione as variáveis", df.columns)

                # Selecionar novo tipo para as variáveis
                new_type = st.selectbox("Selecione o novo tipo", ["object", "int64", "float64"])

                if st.button("Modificar Tipo"):
                    modified_df = df.copy()  # Cria uma cópia do DataFrame original
                    for variable in variables_to_modify:
                        if new_type == "float64" and modified_df[variable].dtype == "object":
                            modified_df[variable] = modified_df[variable].apply(lambda x: x.replace(',', '.') if pd.to_numeric(x, errors='coerce') is None else x).astype(new_type)
                        else:
                            modified_df[variable] = modified_df[variable].astype(new_type)
                    st.session_state.modified_df = modified_df  # Armazena as alterações no DataFrame modificado
                    st.success(f"Tipo das variáveis modificado para '{new_type}'.")
                    st.write(modified_df.dtypes)
                    st.write(modified_df.head())

        

        # Analise Descritiva
        if st.checkbox("🔎Análise Descritiva", key="descriptive_analysis"):
            st.subheader("Análise Descritiva")

            # Verificar se o DataFrame foi modificado interativamente
            if 'modified_df' in st.session_state:
                df = st.session_state.modified_df

            st.write(df.describe())

        # Análise de duplicação
        if st.checkbox("🧰Análise de Variáveis Duplicadas"):
            st.subheader("Análise de Variáveis Duplicadas")
            duplicated_rows = st.session_state.modified_df[st.session_state.modified_df.duplicated()]

            if len(duplicated_rows) > 0:
                st.write(f"Número de linhas duplicadas encontradas: {len(duplicated_rows)}")
                st.write("Linhas duplicadas:")
                st.write(duplicated_rows)
            else:
                st.write("Não foram encontradas variáveis duplicadas.")

            # Opções de tratamento de variáveis duplicadas
            if st.checkbox("🖥️Tratamento de Variáveis Duplicadas"):
                st.subheader("Tratamento de Variáveis Duplicadas")
                treatment_options = ['Excluir Variáveis Duplicadas'] + ['Manter Apenas uma Ocorrência'] * len(duplicated_rows)
                selected_treatment = st.selectbox("Selecione o tratamento desejado", treatment_options)

                # Aplicar tratamento às variáveis duplicadas
                if selected_treatment == 'Excluir Variáveis Duplicadas':
                    st.session_state.modified_df = st.session_state.modified_df.drop_duplicates()
                    st.success("Variáveis duplicadas excluídas.")
                else:
                    st.session_state.modified_df.drop(columns=duplicated_rows, inplace=True)
                    st.success("Apenas uma ocorrência de cada variável duplicada mantida.")

                num_rows_updated, num_cols_updated = st.session_state.modified_df.shape
                num_rows_deleted = num_rows - num_rows_updated

                st.write(st.session_state.modified_df.head())
                col, coll, colll = st.columns(3)
                col.metric("Número de Linhas", f"{num_rows_updated}")
                coll.metric("Número de Colunas", f"{num_cols_updated}")
                colll.metric("Linhas Excluídas", f"{num_rows_deleted}")
                
        # Análise de dados missing
        if st.checkbox("🗄️Análise de Dados Missing"):
            st.subheader("Análise de Dados Missing")
            missing_data = st.session_state.modified_df.isnull().sum().reset_index()
            missing_data.columns = ['Variável', 'Quantidade de Valores Faltantes']
            st.dataframe(missing_data)

            # Opções de tratamento de dados missing
            if st.checkbox("📁Tratamento de Dados Missing"):
                st.subheader("Tratamento de Dados Missing")

                # Tratamento de variáveis numéricas
                numeric_cols = st.session_state.modified_df.select_dtypes(include='number').columns
                numeric_options = ['Excluir Variável'] + ['Substituir pela Média'] * len(numeric_cols)
                numeric_treatment = st.selectbox("Tratamento para Variáveis Numéricas", numeric_options)

                # Tratamento de variáveis categóricas
                categorical_cols = st.session_state.modified_df.select_dtypes(include='object').columns
                categorical_options = ['Excluir Variável'] + ['Substituir pela Moda'] * len(categorical_cols) + ['Excluir Linhas']
                categorical_treatment = st.selectbox("Tratamento para Variáveis Categóricas", categorical_options)

                # Aplicar tratamento aos dados missing
                if numeric_treatment != 'Excluir Variável':
                    st.session_state.modified_df[numeric_cols] = st.session_state.modified_df[numeric_cols].fillna(st.session_state.modified_df[numeric_cols].mean())
                else:
                    st.session_state.modified_df = st.session_state.modified_df.drop(numeric_cols, axis=1)

                if categorical_treatment == 'Substituir pela Moda':
                    st.session_state.modified_df[categorical_cols] = st.session_state.modified_df[categorical_cols].fillna(st.session_state.modified_df[categorical_cols].mode().iloc[0])
                elif categorical_treatment == 'Excluir Linhas':
                    st.session_state.modified_df = st.session_state.modified_df.dropna(subset=categorical_cols)
                else:
                    st.session_state.modified_df = st.session_state.modified_df.drop(categorical_cols, axis=1)

                st.success("Tratamento de dados missing aplicado.")

       

        
        # Detecção e visualização de outliers
        if st.checkbox("📈 Detecção e Visualização de Outliers"):
            st.subheader("Detecção e Visualização de Outliers")

            # Detecção de outliers
            numeric_cols = st.session_state.modified_df.select_dtypes(include='number').columns
            outliers = {}
            for col in numeric_cols:
                q1 = st.session_state.modified_df[col].quantile(0.25)
                q3 = st.session_state.modified_df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers[col] = st.session_state.modified_df[(st.session_state.modified_df[col] < lower_bound) | (st.session_state.modified_df[col] > upper_bound)][col].tolist()

            # Exibição dos outliers
            for col, outlier_values in outliers.items():
                if outlier_values:
                    st.write(f"Outliers encontrados na variável {col}:")
                    df_outliers = pd.DataFrame({'Valor': outlier_values})
                    st.dataframe(df_outliers)

                    # Calcular a média e o desvio padrão da variável
                    mean = df_outliers['Valor'].mean()
                    std = df_outliers['Valor'].std()

                    c1, c2 = st.columns(2)

                    # Exibir a média e o desvio padrão
                    c1.metric(f"Média da variável {col}", f"{mean:.2f}")
                    c2.metric(f"Desvio padrão da variável {col}", f"{std:.2f}")
                else:
                    st.write("")

            # Plot interativo para cada variável
            for col in numeric_cols:
                fig = go.Figure(data=[go.Box(y=st.session_state.modified_df[col], name=col)])
                fig.update_layout(height=500, title=f"Boxplot da variável {col}")
                st.plotly_chart(fig)

            # Opção de tratamento de outliers
            if st.checkbox("🔛Tratamento de Outliers"):
                st.subheader("Tratamento de Outliers")
                selected_variable = st.selectbox("Selecione uma variável para tratar os outliers", numeric_cols)
                treatment_options = ["Remover Outliers", "Substituir por Limites"]
                selected_treatment = st.selectbox("Selecione o tratamento desejado", treatment_options)

                if st.button("🆙 Aplicar Tratamento"):
                    if selected_treatment == "Remover Outliers":
                        q1 = st.session_state.modified_df[selected_variable].quantile(0.25)
                        q3 = st.session_state.modified_df[selected_variable].quantile(0.75)
                        iqr = q3 - q1
                        lower_bound = q1 - 1.5 * iqr
                        upper_bound = q3 + 1.5 * iqr
                        st.session_state.modified_df = st.session_state.modified_df[(st.session_state.modified_df[selected_variable] >= lower_bound) & (st.session_state.modified_df[selected_variable] <= upper_bound)]
                        st.success(f"Outliers removidos da variável {selected_variable}.")

                    elif selected_treatment == "Substituir por Limites":
                        q1 = st.session_state.modified_df[selected_variable].quantile(0.25)
                        q3 = st.session_state.modified_df[selected_variable].quantile(0.75)
                        iqr = q3 - q1
                        lower_bound = q1 - 1.5 * iqr
                        upper_bound = q3 + 1.5 * iqr
                        st.session_state.modified_df[selected_variable] = st.session_state.modified_df[selected_variable].clip(lower_bound, upper_bound)
                        st.success(f"Outliers substituídos por limites na variável {selected_variable}.")

                    num_rows_updated, num_cols_updated = st.session_state.modified_df.shape
                    num_rows_deleted = num_rows - num_rows_updated

                    st.write(st.session_state.modified_df.head())
                    c, cc, ccc = st.columns(3)
                    c.metric("Número de Linhas", f"{num_rows_updated}")
                    cc.metric("Número de Colunas", f"{num_cols_updated}")
                    ccc.metric("Linhas Excluídas", f"{num_rows_deleted}")

        # Redução de dimensionalidade com PCA
        if st.checkbox("🛠️Redução de Dimensionalidade - PCA"):
            st.subheader("Redução de Dimensionalidade - PCA")

            # Seleção das variáveis
            numeric_cols = st.session_state.modified_df.select_dtypes(include='number').columns
            selected_variables = st.multiselect("Selecione as variáveis para aplicar o PCA", numeric_cols)

            if st.button("🔨Aplicar PCA"):
                X = st.session_state.modified_df[selected_variables].values
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(X_scaled)

                df_pca = pd.DataFrame(data=pca_result, columns=["Componente 1", "Componente 2"])
                df_pca["Cluster"] = KMeans(n_clusters=5, random_state=0).fit_predict(X_scaled)

                fig, ax = plt.subplots()
                sns.scatterplot(data=df_pca, x="Componente 1", y="Componente 2", hue="Cluster", palette="Set1", ax=ax)
                ax.set_title("Visualização PCA com Clustering")
                st.pyplot(fig) 

       
        # Correlação
        if st.checkbox("👨‍🔬Matriz de Correlação"):
            st.subheader("Matriz de Correlação")
            corr_matrix = st.session_state.modified_df.corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
            st.pyplot(fig)

            if st.checkbox("📊 Gráfico de Correlação"):
                st.subheader("Gráfico de Correlação")
                corr_fig = px.scatter_matrix(st.session_state.modified_df)
                st.plotly_chart(corr_fig)

            if st.checkbox("🔬Correlação com Variável Alvo"):
                st.subheader("Correlação com Variável Alvo")
                target_variable = st.selectbox("Selecione a variável alvo", st.session_state.modified_df.columns)
                correlation = st.session_state.modified_df.corr()[target_variable].sort_values(ascending=False)
                st.write(correlation)

            # Seleção de variáveis para exclusão
            if st.checkbox("🚮 Seleção de Variáveis para Exclusão"):
                st.subheader("Seleção de Variáveis para Exclusão")
                selected_cols = st.multiselect("Selecione as variáveis para exclusão", st.session_state.modified_df.columns)
                st.session_state.modified_df = st.session_state.modified_df.drop(selected_cols, axis=1)
                st.write("Variáveis selecionadas foram excluídas do conjunto de dados.")

                # Exibir o novo DataFrame
                st.write(st.session_state.modified_df.head())

                # Obter o número de colunas existentes e excluídas
                num_cols = st.session_state.modified_df.shape[1]
                num_cols_deleted = len(selected_cols)

                # Exibir o número de colunas existentes e excluídas
                col1, col2 = st.columns(2)
                col1.metric("Número de Colunas", f"{num_cols}")
                col2.metric("Colunas Excluídas", f"{num_cols_deleted}")
            else:
                st.write("Nenhuma variável selecionada para exclusão.")

        # Transformar variáveis categóricas em dummy
        if st.checkbox("🧮Transformar Variáveis Categóricas em Dummy"):
            st.subheader("Transformar Variáveis Categóricas em Dummy")

            # Selecionar variável categórica para transformar em dummy
            categorical_variable = st.selectbox("Selecione a variável categórica", st.session_state.modified_df.select_dtypes(include='object').columns)

            if st.button("🖱️ Transformar em Dummy"):
                st.session_state.modified_df = pd.get_dummies(st.session_state.modified_df, columns=[categorical_variable], drop_first=True)
                st.success(f"A variável '{categorical_variable}' foi transformada em dummy.")

        #Novo Dataframe
        if st.checkbox("🗃️Criar Novo DataFrame"):
            st.subheader("Criar Novo DataFrame")

            # Exibir lista de colunas
            st.write("Colunas disponíveis:")
            st.write(df.columns)

            # Selecionar coluna para exclusão
            coluna_excluir = st.selectbox("Selecione a coluna para excluir", df.columns)
 
            # Remover coluna selecionada
            novas_colunas = df.columns.tolist()
            novas_colunas.remove(coluna_excluir)

            # Criar novo DataFrame com as colunas restantes
            new_df = pd.DataFrame(columns=novas_colunas)

            # Preencher valores das colunas
            new_data = []
            for col in novas_colunas:
                value = st.text_input(f"Digite o valor para a coluna '{col}'")
                new_data.append(value)

            # Adicionar linha com os valores preenchidos ao novo DataFrame
            new_df.loc[0] = new_data
 
            # Exibir novo DataFrame
            st.write("Novo DataFrame:")
            st.write(new_df)

        #TREINAMENTO
        if st.checkbox("🤖Treinamento do Modelo"):
            st.subheader("Treinamento do Modelo")
            target_variable = st.selectbox("Selecione a variável alvo", options=st.session_state.modified_df.columns, key="target_variable_selectbox")
            clf = setup(data=st.session_state.modified_df, target=target_variable)
            best_model = compare_models()
            st.write("Melhor modelo selecionado:", best_model)

            # Avaliação do modelo selecionado
            metrics_df = pull()
            st.write("📇 Métricas do Modelo:")
            st.dataframe(metrics_df[['Model', 'Accuracy', 'AUC', 'Recall', 'Prec.']])

            # Seleção do modelo
            selected_model = st.selectbox("Selecione um modelo", metrics_df['Model'].tolist())
            best_model = finalize_model(best_model)
            st.write("Modelo selecionado:", selected_model)

        # Previsões
        if st.checkbox("🔮Previsões"):
            st.subheader("Previsões")
            prediction_df = predict_model(best_model)
            st.write(prediction_df.head())

            # Realizar previsão com dados do novo DataFrame
            if st.button("🧙Realizar Previsão"):
                new_prediction = predict_model(best_model, data=new_df)
                st.write("Resultado da Previsão:")
                st.write(new_prediction.head())

                # Adicionar coluna com o valor previsto ao novo DataFrame
                try:
                    new_df['Predição'] = new_prediction['Label']
                    st.write("Novo DataFrame com a coluna de predição:")
                    st.write(new_df)
                except KeyError:
                    st.error("A coluna 'Label' não está presente no DataFrame de previsões.")
                    

                # Exibir novo DataFrame com a coluna de predição
                st.write("Novo DataFrame com a coluna de predição:")
                st.write(new_df)

        # Exportar resultados
        if st.button("📥 Exportar Resultados"):
            export_options = ['CSV', 'Excel']
            selected_export_option = st.selectbox("Selecione o formato de exportação", export_options)

            if selected_export_option == 'CSV':
                csv_file = st.session_state.modified_df.to_csv(index=False)
                st.download_button(
                    label="Baixar arquivo CSV",
                    data=csv_file,
                    file_name="dados_exportados.csv",
                    mime="text/csv"
                )
                st.success("Exportação para CSV concluída.")

            elif selected_export_option == 'Excel':
                excel_file = st.session_state.modified_df.to_excel(index=False)
                st.download_button(
                    label="Baixar arquivo Excel",
                    data=excel_file,
                    file_name="dados_exportados.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                st.success("Exportação para Excel concluída.")

        # Limpar/Resetar
        if st.button("🗑️ Limpar/Resetar"):
            df = None
            st.success("Dados resetados!")
