import streamlit as st
from multistyleseg.data.synthetic import AnnotationType, Task


def global_sidebar():
    with st.sidebar:
        task = st.selectbox(
            "Select Task",
            options=["Color Based", "Texture Based"],
            index=0,
            key="task_selection",
        )
        st.session_state.task = (
            Task.COLOR_BASED if task == "Color Based" else Task.TEXTURE_BASED
        )
        rerun = st.button("Regenerate Synthetic Data")
    if rerun:
        st.rerun()
