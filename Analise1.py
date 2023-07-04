import streamlit as st
import pandas as pd
import pandas_profiling as pp
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import re
import io
import base64
import tempfile

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


if selected3 == "Análise":
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
        
        # Verificar se o usuário deseja gerar um gráfico
        if st.checkbox("📊 Gerar Gráfico de Barras"):
            st.subheader("Gerar Gráfico de Barras")

            # Configurações de estilo do gráfico
            plt.style.use('ggplot')
            bar_color = '#4682B4'
            label_color = '#333333'

            # Selecionar o tipo de gráfico
            chart_type = st.selectbox("Selecione o tipo de gráfico", ['Gráfico de Barras', 'Gráfico de Linhas'])

            # Selecionar a coluna para plotar o gráfico
            column_to_plot = st.selectbox("Selecione a coluna para plotar o gráfico", st.session_state.modified_df.columns)

            # Selecionar a coluna para relacionar (opcional)
            column_to_rel = st.selectbox("Selecione a coluna para relacionar (opcional)", ['', *st.session_state.modified_df.columns])

            # Plotar o gráfico
            if chart_type == 'Gráfico de Barras':
                if column_to_rel:
                    grouped_df = st.session_state.modified_df.groupby(column_to_rel)[column_to_plot].value_counts().unstack()
                    fig = px.bar(grouped_df, x=grouped_df.index, y=grouped_df.columns, color=column_to_rel)
                else:
                    value_counts = st.session_state.modified_df[column_to_plot].value_counts()
                    fig = px.bar(x=value_counts.index, y=value_counts.values)
                    fig.update_layout(
                    xaxis_title=column_to_plot,
                    yaxis_title="Contagem",
                    title="Gráfico de Barras"
                )
            elif chart_type == 'Gráfico de Linhas':
                if column_to_rel:
                    grouped_df = st.session_state.modified_df.groupby(column_to_rel)[column_to_plot].mean().reset_index()
                    fig = px.line(grouped_df, x=column_to_rel, y=column_to_plot, color=column_to_rel, markers=True)
                else:
                    fig = px.line(st.session_state.modified_df, y=column_to_plot, markers=True)
                    fig.update_layout(xaxis_title=column_to_rel if column_to_rel else "Índice",yaxis_title=column_to_plot, title="Gráfico de Linhas")

            # Exibir o gráfico
            st.plotly_chart(fig)
            fig.write_image("grafico.png")
                
               
        #Pandas Profiling
        if st.checkbox("🐼Gerar Relatório de Perfil"):
            profile = pp.ProfileReport(st.session_state.modified_df)
            # Salvar o relatório como arquivo HTML temporário
            with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as temp_file:
                profile.to_file(temp_file.name)
        
            # Exibir o relatório HTML no Streamlit
            with open(temp_file.name, "r") as html_file:
                st.components.v1.html(html_file.read(), height=800)
            
