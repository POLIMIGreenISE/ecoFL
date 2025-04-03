import pandas as pd
import streamlit as st
import numpy as np


def data_visualization(
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
        indexes_train: list,
        indexes_val: list,
        num_clients: int,
        num_clients_names: list = []
):
    '''Visualize distribution for each client'''
    column_name = ["Classes"]
    for i in range(num_clients):
        st.write(f"### Data Distribution Client **{i if not num_clients_names else num_clients_names[i]}**")
        col1, col2 = st.columns(2)
        
        # Training data distribution
        with col1:
            start_index = indexes_train[i]
            end_index = indexes_train[i + 1]
            y_train_df = pd.DataFrame(y_train[start_index:end_index], columns=column_name)
            class_counts = y_train_df[column_name].value_counts().sort_index()
            df_plot = pd.DataFrame({
                "Classes": class_counts.index,
                "Count": class_counts.values
            })
            st.write("#### Train")
            st.divider()
            st.bar_chart(df_plot, x="Classes", y="Count", color='#0000FF')
        
        # Validation data distribution
        with col2:
            start_index = indexes_val[i]
            end_index = indexes_val[i + 1]
            y_val_df = pd.DataFrame(y_val[start_index:end_index], columns=column_name)
            class_counts = y_val_df[column_name].value_counts().sort_index()
            df_plot = pd.DataFrame({
                "Classes": class_counts.index,
                "Count": class_counts.values
            })
            st.write("#### Test")
            st.divider()
            st.bar_chart(df_plot, x="Classes", y="Count", color='#FF0000')

    # Visualize test set distribution
    df_test = pd.DataFrame(y_test, columns=column_name)
    class_counts = df_test[column_name].value_counts().sort_index()
    df_plot = pd.DataFrame({
        "Classes": class_counts.index,
        "Count": class_counts.values
    })
    st.write("### Centralized Test Data Distribution")
    st.divider()
    st.bar_chart(df_plot, x="Classes", y="Count", color='#00FF00')