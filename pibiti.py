import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import streamlit as st
from io import BytesIO  # Necessário para salvar/carregar modelo
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import plotly.graph_objects as go
import joblib  # Necessário para salvar/carregar modelo
import logging
import altair as alt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuração do Streamlit
st.set_page_config(page_title="Monitoramento de Paciente Laplace na UTI Avançado", layout="wide")

# Título e descrição
st.title("Monitoramento Avançado de Pacientes na UTI com Classificação SVM")
st.markdown("""
Este sistema utiliza um classificador SVM para distinguir entre estados 'normais' e 'críticos'.
**Novidades:** Modelo SVM com `class_weight='balanced'`, exibição de probabilidade de criticidade,
ajuste de simulação e resumo estatístico.
""")


# --- PatientMonitor Class (Integrada) ---
class PatientMonitor:
    def __init__(self, tempo_analise=10, min_data_for_train=30):
        self.tempo_analise = tempo_analise
        self.min_data_for_train = min_data_for_train
        self.historico_tempo = []
        self.historico_fc = []
        self.historico_pa_sistolica = []
        self.historico_pa_diastolica = []
        self.historico_temp = []
        self.historico_ox = []
        self.historico_labels = []
        self.predicted_critical_events = []
        self.features = ['frequencia_cardiaca', 'pressao_arterial_sistolica', 'temperatura', 'oxigenacao']

        self.scaler = StandardScaler()
        # MODIFICAÇÃO: Adicionado class_weight='balanced'
        self.svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, class_weight='balanced')
        self.model_trained = False
        self.model_loaded_from_file = False

        self.train_accuracy = None
        self.train_precision = None
        self.train_recall = None
        self.train_f1 = None
        self.train_confusion_matrix = None

    def reset_state(self):
        self.historico_tempo = []
        self.historico_fc = []
        self.historico_pa_sistolica = []
        self.historico_pa_diastolica = []
        self.historico_temp = []
        self.historico_ox = []
        self.historico_labels = []
        self.predicted_critical_events = []
        self.features = ['frequencia_cardiaca', 'pressao_arterial_sistolica', 'temperatura', 'oxigenacao']

        self.scaler = StandardScaler()
        self.svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, class_weight='balanced')
        self.model_trained = False
        self.model_loaded_from_file = False

        self.train_accuracy = None
        self.train_precision = None
        self.train_recall = None
        self.train_f1 = None
        self.train_confusion_matrix = None
        logging.info("PatientMonitor state has been reset.")

    def save_model_to_buffer(self):
        try:
            model_and_scaler = {
                'model': self.svm_model,
                'scaler': self.scaler,
                'features': self.features
            }
            buffer = BytesIO()
            joblib.dump(model_and_scaler, buffer)
            buffer.seek(0)
            logging.info("Modelo e scaler salvos no buffer.")
            return buffer
        except Exception as e:
            logging.error(f"Erro ao salvar modelo no buffer: {e}")
            return None

    def load_model_from_buffer(self, buffer):
        try:
            model_and_scaler = joblib.load(buffer)
            self.svm_model = model_and_scaler['model']
            self.scaler = model_and_scaler['scaler']
            self.features = model_and_scaler.get('features', self.features)
            self.model_trained = True
            self.model_loaded_from_file = True
            self.train_accuracy = None  # Reset metrics as they are from training data
            self.train_precision = None
            self.train_recall = None
            self.train_f1 = None
            self.train_confusion_matrix = None
            logging.info("Modelo e scaler carregados do buffer.")
            return True
        except Exception as e:
            logging.error(f"Erro ao carregar modelo do buffer: {e}")
            self.model_trained = False
            self.model_loaded_from_file = False
            return False

    def predict_patient_state(self, fc, pa_sistolica, temp, ox):
        """Predicts if the current state is critical (1) or normal (0) and its probability"""
        try:
            if not self.model_trained:
                logging.warning("Modelo ainda não treinado. Não é possível prever.")
                return 0, 0.0  # Padrão para normal, probabilidade 0

            novo_dado = pd.DataFrame({
                'frequencia_cardiaca': [fc],
                'pressao_arterial_sistolica': [pa_sistolica],
                'temperatura': [temp],
                'oxigenacao': [ox]
            })
            novo_dado = novo_dado[self.features]

            X_scaled = self.scaler.transform(novo_dado)
            previsao = self.svm_model.predict(X_scaled)[0]
            # MODIFICAÇÃO: Obter probabilidades
            # predict_proba retorna [[prob_classe_0, prob_classe_1]]
            probabilidades = self.svm_model.predict_proba(X_scaled)[0]
            prob_critico = probabilidades[1] if len(
                self.svm_model.classes_) > 1 and 1 in self.svm_model.classes_ else 0.0
            # Ensure class 1 exists before trying to get its probability
            if 1 not in self.svm_model.classes_:  # If only one class seen during training
                prob_critico = 0.0 if self.svm_model.classes_[0] == 0 else 1.0  # if that one class is 0 or 1
                # This case is less likely with class_weight='balanced' and diverse data, but good to handle
            elif len(probabilidades) > 1:  # Normal case with two classes
                prob_critico = probabilidades[list(self.svm_model.classes_).index(1)]

            return previsao, prob_critico
        except Exception as e:
            logging.error(f"Erro ao prever o estado do paciente: {e}")
            return 0, 0.0

    def add_data(self, frequencia_cardiaca, pressao_arterial_sistolica, pressao_arterial_diastolica,
                 temperatura, oxigenacao, current_patient_state):
        try:
            now = pd.Timestamp.now()
            self.historico_tempo.append(now)
            self.historico_fc.append(frequencia_cardiaca)
            self.historico_pa_sistolica.append(pressao_arterial_sistolica)
            self.historico_pa_diastolica.append(pressao_arterial_diastolica)
            self.historico_temp.append(temperatura)
            self.historico_ox.append(oxigenacao)

            label = 1 if current_patient_state == "critical" else 0
            self.historico_labels.append(label)

            if len(self.historico_tempo) >= self.min_data_for_train:
                df_train = pd.DataFrame({
                    'frequencia_cardiaca': self.historico_fc,
                    'pressao_arterial_sistolica': self.historico_pa_sistolica,
                    'temperatura': self.historico_temp,
                    'oxigenacao': self.historico_ox
                })
                X_train = df_train[self.features]
                y_train = pd.Series(self.historico_labels)

                # Check if y_train has more than one class before training
                if len(np.unique(
                        y_train)) < 2 and self.model_loaded_from_file == False:  # Only warn if not using a loaded model
                    logging.warning(
                        f"Tentando treinar o modelo, mas y_train tem apenas {len(np.unique(y_train))} classe(s) únicas. O SVM precisa de pelo menos duas classes para treinar efetivamente e calcular probabilidades. Coletando mais dados diversos...")
                    # Reset metrics as they would be invalid
                    self.train_accuracy = None
                    self.train_precision = None
                    self.train_recall = None
                    self.train_f1 = None
                    self.train_confusion_matrix = None
                    # self.model_trained = False # Keep it true if it was loaded, otherwise training might not happen
                else:
                    X_scaled = self.scaler.fit_transform(X_train)
                    self.svm_model.fit(X_scaled, y_train)
                    self.model_trained = True  # Mark as trained after successful fit
                    self.model_loaded_from_file = False  # Any training means it's not just the loaded model

                    y_pred_train = self.svm_model.predict(X_scaled)
                    self.train_accuracy = accuracy_score(y_train, y_pred_train)
                    # Use zero_division=0 for cases where a class might not be present in predictions
                    # Ensure labels=[0, 1] for metrics when possible
                    unique_labels_train = np.unique(y_train)
                    unique_labels_pred = np.unique(y_pred_train)

                    if 1 in unique_labels_train or 1 in unique_labels_pred:  # Calculate if class 1 is present in true or pred
                        self.train_precision = precision_score(y_train, y_pred_train, pos_label=1, zero_division=0)
                        self.train_recall = recall_score(y_train, y_pred_train, pos_label=1, zero_division=0)
                        self.train_f1 = f1_score(y_train, y_pred_train, pos_label=1, zero_division=0)
                    else:  # Class 1 not present, metrics for it are undefined or 0
                        self.train_precision = 0.0
                        self.train_recall = 0.0
                        self.train_f1 = 0.0

                    try:
                        self.train_confusion_matrix = confusion_matrix(y_train, y_pred_train, labels=[0, 1])
                    except ValueError:
                        self.train_confusion_matrix = None
                        logging.warning("Não foi possível calcular a matriz de confusão.")
                    logging.info(f"Modelo (re)treinado. Acurácia de Treinamento: {self.train_accuracy:.2f}")

            if self.model_trained:
                # MODIFICAÇÃO: predict_patient_state agora retorna previsão e probabilidade
                is_critical_prediction, prob_critical_prediction = self.predict_patient_state(
                    frequencia_cardiaca, pressao_arterial_sistolica, temperatura, oxigenacao
                )
                if is_critical_prediction == 1:
                    self.predicted_critical_events.append({
                        'timestamp': now,
                        'frequencia_cardiaca': frequencia_cardiaca,
                        'pressao_arterial_sistolica': pressao_arterial_sistolica,
                        'pressao_arterial_diastolica': pressao_arterial_diastolica,
                        'temperatura': temperatura,
                        'oxigenacao': oxigenacao,
                        'true_label_simulated': label,
                        'predicted_label': is_critical_prediction,
                        'probability_critical': prob_critical_prediction  # MODIFICAÇÃO: Salvar probabilidade
                    })
                    logging.warning(
                        f"Classificador previu um evento CRÍTICO (Prob: {prob_critical_prediction:.2f})! (Estado simulado: {current_patient_state})")

            max_len = self.tempo_analise * 60
            if len(self.historico_tempo) > max_len:
                self.historico_tempo = self.historico_tempo[-max_len:]
                self.historico_fc = self.historico_fc[-max_len:]
                self.historico_pa_sistolica = self.historico_pa_sistolica[-max_len:]
                self.historico_pa_diastolica = self.historico_pa_diastolica[-max_len:]
                self.historico_temp = self.historico_temp[-max_len:]
                self.historico_ox = self.historico_ox[-max_len:]
                self.historico_labels = self.historico_labels[-max_len:]

        except Exception as e:
            logging.error(f"Erro ao adicionar dados: {e}", exc_info=True)

    def get_dataframe(self):
        try:
            return pd.DataFrame({
                'timestamp': self.historico_tempo,
                'frequencia_cardiaca': self.historico_fc,
                'pressao_arterial_sistolica': self.historico_pa_sistolica,
                'pressao_arterial_diastolica': self.historico_pa_diastolica,
                'temperatura': self.historico_temp,
                'oxigenacao': self.historico_ox,
                'label_simulated': self.historico_labels
            })
        except Exception as e:
            logging.error(f"Erro ao criar dataframe: {e}")
            return pd.DataFrame()

    def get_predicted_critical_events_dataframe(self):
        return pd.DataFrame(self.predicted_critical_events)


# --- Vital Sign Simulation (Integrada) ---
# MODIFICAÇÃO: Parâmetros de simulação como argumentos com valores padrão
def simular_sinais_vitais(state="stable", critical_fc_mean=110):  # critical_fc_mean adicionado
    if state == "stable":
        fc_mean, fc_std = 75, 5
        pa_sist_mean, pa_sist_std = 120, 8
        temp_mean, temp_std = 36.7, 0.2
        ox_mean, ox_std = 98, 0.5
    elif state == "critical":
        # MODIFICAÇÃO: Usa o critical_fc_mean fornecido
        fc_mean, fc_std = critical_fc_mean, 10
        pa_sist_mean, pa_sist_std = 140, 15
        temp_mean, temp_std = 38.5, 0.5
        ox_mean, ox_std = 92, 2
    else:  # improving
        fc_mean, fc_std = 70, 3
        pa_sist_mean, pa_sist_std = 110, 5
        temp_mean, temp_std = 36.6, 0.1
        ox_mean, ox_std = 99, 0.3

    fc = np.random.normal(fc_mean, fc_std)
    pa_sistolica = np.random.normal(pa_sist_mean, pa_sist_std)
    temp = np.random.normal(temp_mean, temp_std)
    ox = np.random.normal(ox_mean, ox_std)
    pa_diastolica = pa_sistolica - np.random.normal(40, 5)

    current_time_effect = time.time()
    fc += np.sin(current_time_effect / 10) * 3
    pa_sistolica += np.cos(current_time_effect / 12) * 5
    pa_diastolica += np.cos(current_time_effect / 12) * 3
    temp += np.sin(current_time_effect / 20) * 0.1
    ox += np.cos(current_time_effect / 15) * 0.5

    fc = max(30, min(fc, 200))
    pa_sistolica = max(60, min(pa_sistolica, 220))
    pa_diastolica = max(40, min(pa_diastolica, 140))
    if pa_sistolica <= pa_diastolica:
        pa_sistolica = pa_diastolica + np.random.uniform(10, 20)
    temp = max(34, min(temp, 41))
    ox = max(80, min(ox, 100))
    return fc, pa_sistolica, pa_diastolica, temp, ox


# --- Sidebar for User Controls ---
with st.sidebar:
    st.header("Controles Interativos")
    patient_state_input = st.selectbox(
        "Estado do Paciente (Simulação/Treinamento):",
        ["stable", "critical", "improving"], key="patient_state_selector"
    )
    tempo_analise_input = st.slider(
        "Intervalo de Tempo (minutos) para Visualização:",
        min_value=1, max_value=60, value=10, key="tempo_analise_slider"
    )

    st.markdown("---")
    st.subheader("Ajustes Finos da Simulação")
    # MODIFICAÇÃO: Slider para ajustar a média da FC no estado crítico
    critical_fc_mean_input = st.slider(
        "Média FC (Estado Crítico):",
        min_value=90, max_value=150, value=110, key="critical_fc_mean_slider"
    )

    st.markdown("---")
    st.header("Gerenciamento de Dados")
    if st.button("Exportar Dados Históricos"):
        if 'df_historico' in st.session_state and not st.session_state.get('df_historico', pd.DataFrame()).empty:
            csv = st.session_state['df_historico'].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Baixar CSV Histórico", data=csv, file_name="dados_historicos_paciente.csv", mime="text/csv"
            )
        else:
            st.warning("Nenhum dado histórico para exportar.")

    if st.button("Exportar Eventos Críticos Previstos"):
        if 'df_critical_events' in st.session_state and not st.session_state.get('df_critical_events',
                                                                                 pd.DataFrame()).empty:
            csv_critical = st.session_state['df_critical_events'].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Baixar CSV Eventos Críticos", data=csv_critical, file_name="eventos_criticos_previstos.csv",
                mime="text/csv"
            )
        else:
            st.warning("Nenhum evento crítico previsto para exportar.")

    st.markdown("---")
    st.header("Gerenciamento do Modelo SVM")
    if st.button("Salvar Modelo SVM Treinado"):
        if 'patient_monitor' in st.session_state and st.session_state['patient_monitor'].model_trained:
            monitor_instance = st.session_state['patient_monitor']
            model_buffer = monitor_instance.save_model_to_buffer()
            if model_buffer:
                st.download_button(
                    label="Baixar Arquivo do Modelo (.joblib)",
                    data=model_buffer,
                    file_name="svm_patient_monitor_model.joblib",
                    mime="application/octet-stream"
                )
                st.success("Modelo SVM e Scaler prontos para download.")
        else:
            st.warning("Modelo ainda não treinado ou não disponível para salvar.")

    uploaded_model_file = st.file_uploader("Carregar Modelo SVM (.joblib)", type=["joblib"], key="model_uploader")
    if uploaded_model_file is not None:
        if 'patient_monitor' in st.session_state:
            monitor_instance = st.session_state['patient_monitor']
            if monitor_instance.load_model_from_buffer(uploaded_model_file):
                st.success("Modelo SVM e Scaler carregados com sucesso! O monitor usará este modelo.")
                st.session_state.model_uploader = None  # Reset uploader
                st.experimental_rerun()  # Rerun to reflect changes and clear uploader
            else:
                st.error("Erro ao carregar o modelo. Verifique o arquivo ou console para detalhes.")
        # else: # This case should ideally not happen if patient_monitor is always in session_state
        # st.session_state['patient_monitor'] = PatientMonitor() # Initialize if somehow lost
        # st.warning("Monitor reinicializado. Tente carregar o modelo novamente.")

    if st.button("Resetar Modelo e Dados Históricos"):
        if 'patient_monitor' in st.session_state:
            st.session_state['patient_monitor'].reset_state()
            st.success("Modelo e dados históricos resetados. O sistema recomeçará a coleta e treinamento.")
            st.experimental_rerun()

    st.markdown("---")
    st.header("Informações do Paciente")
    st.markdown("**Nome:** Laplace")
    st.markdown("**ID:** LP777")
    st.markdown("---")
    st.markdown("**Informações do Sistema**")
    st.markdown(f"""
        - **Modelo:** Classificação SVM (`class_weight='balanced'`)
        - **Versão:** 2.2.1 (Unificado)
        - **Última Atualização:** {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M:%S')}
        - **Desenvolvido por:** Equipe de Análise de Dados
    """)

# --- Initialize PatientMonitor ---
if 'patient_monitor' not in st.session_state:
    st.session_state['patient_monitor'] = PatientMonitor(tempo_analise=tempo_analise_input)
    logging.info("Nova instância do PatientMonitor criada.")

monitor = st.session_state['patient_monitor']
monitor.tempo_analise = tempo_analise_input

# --- Placeholders for Metrics and Charts ---
placeholder_metricas = st.empty()
placeholder_model_status = st.empty()
placeholder_model_performance = st.empty()
placeholder_grafico_resumo = st.empty()

# --- Main Loop for Real-Time Updates ---
while True:
    try:
        # MODIFICAÇÃO: Passar critical_fc_mean_input para a simulação
        fc_sim, pa_sist_sim, pa_diast_sim, temp_sim, ox_sim = simular_sinais_vitais(
            patient_state_input,
            critical_fc_mean=critical_fc_mean_input  # Usa o valor do slider
        )

        monitor.add_data(fc_sim, pa_sist_sim, pa_diast_sim, temp_sim, ox_sim, patient_state_input)

        df_historico = monitor.get_dataframe()
        df_critical_events = monitor.get_predicted_critical_events_dataframe()

        st.session_state['df_historico'] = df_historico
        st.session_state['df_critical_events'] = df_critical_events

        with placeholder_metricas.container():
            col1, col2, col3, col4 = st.columns(4)
            if not df_historico.empty:
                latest_fc = df_historico['frequencia_cardiaca'].iloc[-1]
                latest_pa_sist = df_historico['pressao_arterial_sistolica'].iloc[-1]
                latest_pa_diast = df_historico['pressao_arterial_diastolica'].iloc[-1]
                latest_temp = df_historico['temperatura'].iloc[-1]
                latest_ox = df_historico['oxigenacao'].iloc[-1]

                delta_fc = latest_fc - (
                    df_historico['frequencia_cardiaca'].iloc[-2] if len(df_historico) > 1 else latest_fc)
                delta_pa_sistolica = latest_pa_sist - (
                    df_historico['pressao_arterial_sistolica'].iloc[-2] if len(df_historico) > 1 else latest_pa_sist)
                delta_pa_diastolica = latest_pa_diast - (
                    df_historico['pressao_arterial_diastolica'].iloc[-2] if len(df_historico) > 1 else latest_pa_diast)
                delta_temp = latest_temp - (
                    df_historico['temperatura'].iloc[-2] if len(df_historico) > 1 else latest_temp)
                delta_ox = latest_ox - (df_historico['oxigenacao'].iloc[-2] if len(df_historico) > 1 else latest_ox)

                with col1:
                    st.metric("Frequência Cardíaca", f"{latest_fc:.1f} bpm", delta=f"{delta_fc:.1f}")
                with col2:
                    st.metric("Pressão Arterial", f"{latest_pa_sist:.1f}/{latest_pa_diast:.1f} mmHg",
                              delta=f"{delta_pa_sistolica:.1f}/{delta_pa_diastolica:.1f}")
                with col3:
                    st.metric("Temperatura", f"{latest_temp:.1f} °C", delta=f"{delta_temp:.1f}")
                with col4:
                    st.metric("Oxigenação", f"{latest_ox:.1f} %", delta=f"{delta_ox:.1f}")
            else:
                # Para evitar que as colunas colapsem quando vazias
                with col1:
                    st.empty()
                with col2:
                    st.empty()
                with col3:
                    st.empty()
                with col4:
                    st.empty()

        with placeholder_model_status.container():
            # Alerta de Evento Crítico
            if not df_critical_events.empty and \
                    not df_historico.empty and \
                    df_critical_events['timestamp'].iloc[-1] == df_historico['timestamp'].iloc[-1]:
                last_event = df_critical_events.iloc[-1]
                prob_crit = last_event.get('probability_critical', 0.0)
                st.toast(f"🚨 ALERTA: Evento CRÍTICO previsto! (Prob: {prob_crit:.0%})", icon="🚨")
                st.error(
                    f"ALERTA: Modelo SVM previu um evento CRÍTICO! (Probabilidade: {prob_crit:.0%}) (Simulação atual: {patient_state_input})")
            # Status do Modelo
            elif monitor.model_loaded_from_file:
                st.success("Modelo SVM carregado de arquivo e ativo. Previsões em andamento.")
            elif monitor.model_trained:
                st.success("Sinais vitais monitorados. Modelo SVM treinado e ativo. Previsões em andamento.")
            else:  # Coletando dados para o primeiro treinamento
                data_collected = len(df_historico) if not df_historico.empty else 0
                st.info(
                    f"Coletando dados para treinar o modelo SVM... ({data_collected}/{monitor.min_data_for_train} pontos)")

        with placeholder_model_performance.container():
            # Exibir métricas se o modelo foi treinado/retreinado NESTA SESSÃO E tem métricas calculadas
            if monitor.model_trained and not monitor.model_loaded_from_file and monitor.train_accuracy is not None:
                st.subheader("Desempenho do Modelo SVM (nos dados de treinamento atuais)")
                col_perf1, col_perf2, col_perf3, col_perf4 = st.columns(4)
                col_perf1.metric("Acurácia (Treino)", f"{monitor.train_accuracy:.2%}")
                col_perf2.metric("Precisão Crítico (Treino)",
                                 f"{monitor.train_precision:.2%}" if monitor.train_precision is not None else "N/A")
                col_perf3.metric("Recall Crítico (Treino)",
                                 f"{monitor.train_recall:.2%}" if monitor.train_recall is not None else "N/A")
                col_perf4.metric("F1-Score Crítico (Treino)",
                                 f"{monitor.train_f1:.2%}" if monitor.train_f1 is not None else "N/A")

                if monitor.train_confusion_matrix is not None:
                    with st.expander("Ver Matriz de Confusão (Treinamento)", expanded=False):
                        fig_cm, ax = plt.subplots(figsize=(4, 3))
                        sns.heatmap(monitor.train_confusion_matrix, annot=True, fmt='d', cmap='Blues',
                                    xticklabels=['Prev. Normal', 'Prev. Crítico'],
                                    yticklabels=['Real Normal', 'Real Crítico'], ax=ax)
                        ax.set_xlabel("Previsão do Modelo")
                        ax.set_ylabel("Rótulo Real (Simulado)")
                        st.pyplot(fig_cm)
            # Mensagem para modelo carregado (que ainda não foi retreinado com novos dados para mostrar métricas)
            elif monitor.model_loaded_from_file and not (
                    monitor.model_trained and not monitor.model_loaded_from_file and monitor.train_accuracy is not None):
                st.info(
                    "Modelo SVM carregado. Métricas de desempenho serão exibidas após retreinamento com novos dados.")
            # Mensagem para modelo ainda não treinado de forma alguma
            elif not monitor.model_trained:
                st.info("Modelo SVM ainda não treinado. Métricas de desempenho serão exibidas após o treinamento.")

        with placeholder_grafico_resumo.container():
            if not df_historico.empty:
                st.subheader("Análise dos Sinais Vitais")
                tab_graficos, tab_resumo, tab_eventos = st.tabs([
                    "📈 Gráficos dos Sinais Vitais",
                    "📊 Resumo Estatístico",
                    "🚨 Eventos Críticos Previstos"
                ])


                def add_critical_event_markers(fig, critical_df, y_col_name):
                    if not critical_df.empty:
                        hover_texts = [f"Prob. Crítico: {prob:.0%}" for prob in
                                       critical_df.get('probability_critical', [0.0] * len(critical_df))]
                        fig.add_trace(go.Scatter(
                            x=critical_df['timestamp'], y=critical_df[y_col_name],
                            mode='markers', marker=dict(color='magenta', size=12, symbol='x'),
                            name='Evento Crítico Previsto',
                            text=hover_texts,
                            hoverinfo='x+y+text'
                        ))
                    return fig


                with tab_graficos:
                    col_g1, col_g2 = st.columns(2)
                    with col_g1:
                        fig_fc = go.Figure()
                        fig_fc.add_trace(
                            go.Scatter(x=df_historico['timestamp'], y=df_historico['frequencia_cardiaca'], mode='lines',
                                       name='FC'))
                        fig_fc = add_critical_event_markers(fig_fc, df_critical_events, 'frequencia_cardiaca')
                        fig_fc.update_layout(title='Frequência Cardíaca', yaxis_title='bpm', height=350,
                                             margin=dict(t=30, b=5, l=5, r=5))
                        st.plotly_chart(fig_fc, use_container_width=True)

                        fig_temp = go.Figure()
                        fig_temp.add_trace(
                            go.Scatter(x=df_historico['timestamp'], y=df_historico['temperatura'], mode='lines',
                                       name='Temp.'))
                        fig_temp = add_critical_event_markers(fig_temp, df_critical_events, 'temperatura')
                        fig_temp.update_layout(title='Temperatura Corporal', yaxis_title='°C', height=350,
                                               margin=dict(t=30, b=5, l=5, r=5))
                        st.plotly_chart(fig_temp, use_container_width=True)
                    with col_g2:
                        fig_pa = go.Figure()
                        fig_pa.add_trace(
                            go.Scatter(x=df_historico['timestamp'], y=df_historico['pressao_arterial_sistolica'],
                                       mode='lines', name='Sistólica', line=dict(color='red')))
                        fig_pa.add_trace(
                            go.Scatter(x=df_historico['timestamp'], y=df_historico['pressao_arterial_diastolica'],
                                       mode='lines', name='Diastólica', line=dict(color='blue')))
                        fig_pa = add_critical_event_markers(fig_pa, df_critical_events, 'pressao_arterial_sistolica')
                        fig_pa.update_layout(title='Pressão Arterial', yaxis_title='mmHg', height=350,
                                             margin=dict(t=30, b=5, l=5, r=5))
                        st.plotly_chart(fig_pa, use_container_width=True)

                        fig_ox = go.Figure()
                        fig_ox.add_trace(
                            go.Scatter(x=df_historico['timestamp'], y=df_historico['oxigenacao'], mode='lines',
                                       name='SpO2'))
                        fig_ox = add_critical_event_markers(fig_ox, df_critical_events, 'oxigenacao')
                        fig_ox.update_layout(title='Saturação de Oxigênio', yaxis_title='%', height=350,
                                             margin=dict(t=30, b=5, l=5, r=5))
                        st.plotly_chart(fig_ox, use_container_width=True)

                with tab_resumo:
                    st.subheader("Resumo Estatístico dos Dados Históricos Coletados")
                    df_numeric_historico = df_historico[
                        ['frequencia_cardiaca', 'pressao_arterial_sistolica', 'pressao_arterial_diastolica',
                         'temperatura', 'oxigenacao']]
                    if not df_numeric_historico.empty:
                        st.dataframe(df_numeric_historico.describe().T.style.format("{:.2f}"), use_container_width=True)
                    else:
                        st.info("Dados insuficientes para o resumo estatístico.")

                with tab_eventos:
                    st.subheader("Detalhes dos Eventos Críticos Previstos pelo Modelo SVM")
                    if not df_critical_events.empty:
                        display_df_critical = df_critical_events[['timestamp', 'frequencia_cardiaca',
                                                                  'pressao_arterial_sistolica', 'temperatura',
                                                                  'oxigenacao',
                                                                  'true_label_simulated',
                                                                  'probability_critical']].copy()  # Use .copy() to avoid SettingWithCopyWarning
                        display_df_critical['true_label_simulated'] = display_df_critical['true_label_simulated'].map(
                            {0: 'Normal/Improving (Sim.)', 1: 'Critical (Sim.)'})
                        display_df_critical['probability_critical_fmt'] = display_df_critical[
                            'probability_critical'].map('{:.0%}'.format)
                        display_df_critical = display_df_critical.sort_values(by='timestamp', ascending=False)

                        st.dataframe(
                            display_df_critical[['timestamp', 'frequencia_cardiaca', 'pressao_arterial_sistolica',
                                                 'temperatura', 'oxigenacao', 'true_label_simulated',
                                                 'probability_critical_fmt']].rename(
                                columns={'probability_critical_fmt': 'Prob. Crítico'}),
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "timestamp": st.column_config.DatetimeColumn("Timestamp", format="DD/MM/YY HH:mm:ss"),
                                "frequencia_cardiaca": st.column_config.NumberColumn("FC", format="%.1f bpm"),
                                "pressao_arterial_sistolica": st.column_config.NumberColumn("PAS", format="%.1f mmHg"),
                                "temperatura": st.column_config.NumberColumn("Temp.", format="%.1f °C"),
                                "oxigenacao": st.column_config.NumberColumn("SpO2", format="%.1f %%"),
                                # Use %% for literal %
                            }
                        )

                        st.write("#### Relação Entre Sinais em Eventos Críticos Previstos")
                        if len(df_critical_events) > 1:
                            altair_chart = alt.Chart(df_critical_events).mark_circle(size=100).encode(
                                x=alt.X('frequencia_cardiaca:Q', title='FC (bpm)'),
                                y=alt.Y('pressao_arterial_sistolica:Q', title='PAS (mmHg)'),
                                color=alt.Color('temperatura:Q', title='Temp (°C)', scale=alt.Scale(scheme='viridis')),
                                size=alt.Size('oxigenacao:Q', title='SpO2 (%)'),
                                tooltip=[alt.Tooltip('timestamp:T', title='Timestamp'),
                                         alt.Tooltip('frequencia_cardiaca:Q', title='FC', format='.1f'),
                                         alt.Tooltip('pressao_arterial_sistolica:Q', title='PAS', format='.1f'),
                                         alt.Tooltip('temperatura:Q', title='Temp.', format='.1f'),
                                         alt.Tooltip('oxigenacao:Q', title='SpO2', format='.1f'),
                                         alt.Tooltip('true_label_simulated:N', title='Simulado'),
                                         alt.Tooltip('probability_critical:Q', title='Prob. Crítico', format='.0%')]
                            ).interactive()
                            st.altair_chart(altair_chart, use_container_width=True)
                        else:
                            st.info(
                                "Dados insuficientes para gráfico de dispersão interativo (mínimo 2 eventos críticos).")
                    else:
                        st.info("Nenhum evento crítico previsto pelo modelo SVM até o momento.")
            else:
                st.info("Aguardando dados para exibir gráficos e resumo...")

    except Exception as e:
        logging.error(f"Erro no loop principal: {e}", exc_info=True)
        st.error(f"Ocorreu um erro catastrófico no loop principal: {e}")
        # Consider breaking the loop or having a more robust restart mechanism for production
        break  # For this example, break on unhandled error in main loop

    time.sleep(1)