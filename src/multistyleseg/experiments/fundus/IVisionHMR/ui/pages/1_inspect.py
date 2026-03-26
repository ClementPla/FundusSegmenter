import streamlit as st
from multistyleseg.analysis.dr_grading.hmr_diagnosis import (
    create_diagnosis,
    get_macular_severity,
    standardize_macular_threat,
    get_retinopathy_severity,
    standardize_retinopathy,
)
import pandas as pd
from pathlib import Path
from multistyleseg.data.fundus.consts import Lesions, ALL_CLASSES
import numpy as np
import matplotlib.pyplot as plt
import cv2
from fundus_data_toolkit.utils.composer import get_generic_composer
import plotly.graph_objects as go
import plotly.express as px

from PIL import Image

st.set_page_config(layout="wide", page_title="IVisionHMR Inspection")


root_inference = Path(
    "/home/clement/Documents/Projets/MultiStyleSeg/src/multistyleseg/experiments/fundus/IVisionHMR/ensemble_inference"
)

root_macula = Path(
    "/home/clement/Documents/Projets/MultiStyleSeg/src/multistyleseg/experiments/fundus/IVisionHMR/od_mac"
)


@st.cache_data(show_spinner="Loading datasets...")
def load_dfs():
    root_gt = Path("/home/clement/Documents/data/IVisionHMR/metadata")

    df_male = pd.read_csv(root_gt / "male.csv")
    df_female = pd.read_csv(root_gt / "female.csv")
    df_gt = pd.concat([df_male, df_female], ignore_index=True)

    all_pickles_files = list(root_inference.glob("*.pkl"))

    all_pickles_files_macula = list(root_macula.glob("*.pkl"))
    # Read and concatenate all pickle files
    df_inference = pd.concat(
        [pd.read_pickle(pkl_file) for pkl_file in all_pickles_files], ignore_index=True
    )
    df_od_mac = pd.concat(
        [pd.read_pickle(pkl_file) for pkl_file in all_pickles_files_macula],
        ignore_index=True,
    )
    df_od_mac.od_valid = df_od_mac.od_valid.apply(lambda x: bool(x))
    df_od_mac.macula_valid = df_od_mac.macula_valid.apply(lambda x: bool(x))
    return df_gt, df_inference, df_od_mac


COLS_DIAGNOSIS = [
    "Rétinopathie diabétique ",
    "Menace diabétique de la macula ",
    "DMLA ",
]


@st.cache_data(show_spinner=False)
def get_results(df_inference, df_od_mac, df_gt):
    # For image_id in df_inference, look at the corresponding diagnosis in df_gt
    results = []
    progress_bar = st.progress(0, text="Computing Results...")
    total = len(df_inference["image_id"].unique())
    for idx, image_id in enumerate(df_inference["image_id"].unique()):
        progress_bar.progress((idx + 1) / total)
        for laterality in ["OS", "OD"]:
            laterality_columns = [col + laterality for col in COLS_DIAGNOSIS]
            gt = df_gt[df_gt["No session"] == int(image_id)][laterality_columns]
            # We rename the columns to remove laterality
            gt = gt.rename(
                columns={
                    col: col.replace(f" {laterality}", "") for col in laterality_columns
                }
            )
            pred = df_inference[
                (df_inference["image_id"] == image_id)
                & (df_inference["laterality"] == laterality)
            ]
            gt["session_id"] = image_id
            gt["laterality"] = laterality
            od_mac_row = df_od_mac[
                (df_od_mac["image_id"] == image_id)
                & (df_od_mac["laterality"] == laterality)
            ]
            for lesion in ALL_CLASSES:
                pred_lesion = pred[pred.lesion_id == lesion.name]
                gt["N " + lesion.name] = len(pred_lesion)
            for lesion in ALL_CLASSES:
                # For each lesion, we check if within 100 pixels of the macula center
                # If yes, we consider it as threatening the macula
                threatening_lesions = 0
                for _, row in pred[pred.lesion_id == lesion.name].iterrows():
                    if not bool(od_mac_row.iloc[0].macula_valid):
                        threatening_lesions = np.nan
                        break
                    macula_x, macula_y = tuple(od_mac_row.iloc[0]["macula"])
                    lesion_y, lesion_x = row["centroid"]
                    distance = np.sqrt(
                        (macula_x - lesion_x) ** 2 + (macula_y - lesion_y) ** 2
                    )
                    if distance <= 250 and bool(od_mac_row.iloc[0]["macula_valid"]):
                        threatening_lesions += 1
                gt[f"N {lesion.name} threatening macula"] = threatening_lesions
            results.append(gt)
    return results


@st.cache_data(persist="disk", show_spinner=False)
def get_df():
    df_gt, df_inference, df_od_mac = load_dfs()

    results = get_results(df_inference, df_od_mac, df_gt)

    df = pd.concat(results, ignore_index=True)
    df["N Exudates, Hemorrhages, Microaneurysms"] = df[
        ["N " + lesion.name for lesion in [Lesions.HEMORRHAGES, Lesions.MICROANEURYSMS]]
    ].sum(axis=1)
    df["N Significant"] = df["N Exudates, Hemorrhages, Microaneurysms"]
    # Standardize columns
    df["Retinopathy_std"] = df["Rétinopathie diabétique"].apply(standardize_retinopathy)
    df["Macular_std"] = df["Menace diabétique de la macula"].apply(
        standardize_macular_threat
    )

    # Add severity scores for sorting
    df["Retino_severity"] = df["Retinopathy_std"].apply(get_retinopathy_severity)
    df["Macular_severity"] = df["Macular_std"].apply(get_macular_severity)

    # Create combined category (C3) and diagnosis
    df["C3"] = df["Retinopathy_std"] + " | " + df["Macular_std"]
    df["Diagnosis"] = df.apply(create_diagnosis, axis=1)
    return df


df = get_df().copy()

with st.sidebar:
    lesions_choice = st.segmented_control(
        "Select lesions to display counts for:",
        options=[lesion.name for lesion in ALL_CLASSES],
        default=[Lesions.MICROANEURYSMS.name],
        selection_mode="multi",
    )
    and_or_or = st.radio(
        "Combine lesion counts with:",
        options=["OR", "AND"],
        index=0,
        horizontal=True,
    )
    bins_choice = st.selectbox(
        "Select bins to display:",
        options=["All", "No lesions", "1+", "1-5", "6-20", "21-50", "51+"],
        index=0,
    )
    actual_diagnosis_choice = st.selectbox(
        "Select actual diagnosis to check:",
        options=["All", "Any DR"] + sorted(df["Diagnosis"].unique().tolist()),
        index=1,
    )

    # Filter df based on bins choice and lesions choice
    if bins_choice != "All":
        bin_ranges = {
            "No lesions": (0, 0),
            "1+": (1, float("inf")),
            "1-5": (1, 5),
            "6-20": (6, 20),
            "21-50": (21, 50),
            "51+": (51, float("inf")),
        }
        min_bin, max_bin = bin_ranges[bins_choice]

        mask = (
            pd.Series(True, index=df.index)
            if and_or_or == "AND"
            else pd.Series(False, index=df.index)
        )

        for lesion in lesions_choice:
            lesion_counts = df["N " + lesion]
            lesion_mask = lesion_counts.between(min_bin, max_bin)
            if and_or_or == "OR":
                mask = mask | lesion_mask
            else:  # AND
                mask = mask & lesion_mask
        df = df[mask]
    if actual_diagnosis_choice != "All":
        if actual_diagnosis_choice == "Any DR":
            # We keep the diagnosis from 04 to 13
            diagnosis_columns = [
                f
                for f in df["Diagnosis"].unique()
                if int(f.split("- ")[0]) >= 4 and int(f.split("- ")[0]) <= 13
            ]
            df = df[df["Diagnosis"].isin(diagnosis_columns)]
        else:
            df = df[df["Diagnosis"] == actual_diagnosis_choice]


df.sort_values(
    by=["Retino_severity", "Macular_severity"],
    ascending=False,
    inplace=True,
)

selected = st.dataframe(
    df,
    on_select="rerun",
    hide_index=True,
)
composer = get_generic_composer(shape=(1536, 1536))
composer.deactivate_op(2)
composer.deactivate_op(3)
ROOT_IMG = Path("/home/clement/Documents/data/IVisionHMR/output/fundus/")
# For selected rows, show images and inference results
if selected.selection:
    for i in selected["selection"]["rows"]:
        row = df.iloc[i]
        st.subheader(
            f"Session {row['session_id']} - {row['laterality']} - Diagnosis: {row['Diagnosis']}"
        )
        img_path = (
            ROOT_IMG
            / str(int(row["session_id"]))
            / row["laterality"]
            / "stitching.jpeg"
        )
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR_RGB)

        # Load the inference pickle
        inference_pkl = (
            root_inference / f"{int(row['session_id'])}_{row['laterality']}_lesions.pkl"
        )

        df_inference = pd.read_pickle(inference_pkl)
        img = composer(image=img)["image"]
        fig = go.Figure()

        # 2. Add the background image
        # Plotly uses a coordinate system where (0,0) is often the top-left for images
        img_height, img_width = img.shape[:2]

        fig.add_layout_image(
            dict(
                source=Image.fromarray(img),  # Convert NumPy array to PIL Image
                xref="x",
                yref="y",
                x=0,
                y=0,
                sizex=img_width,
                sizey=img_height,
                sizing="stretch",
                opacity=1,
                layer="below",
            )
        )

        # 3. Add scatter traces for each lesion type
        for lesion in ALL_CLASSES:
            lesion_rows = df_inference[df_inference.lesion_id == lesion.name]
            if not lesion_rows.empty:
                ys = lesion_rows["centroid"].apply(lambda x: x[0]).values
                xs = lesion_rows["centroid"].apply(lambda x: x[1]).values

                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="markers",
                        name=lesion.name,
                        marker=dict(size=8, opacity=0.7),
                        hovertemplate=f"<b>{lesion.name}</b><br>X: %{{x}}<br>Y: %{{y}}<extra></extra>",
                    )
                )

        # 4. Configure axes and layout
        fig.update_xaxes(showgrid=False, range=[0, img_width], visible=False)
        fig.update_yaxes(
            showgrid=False,
            range=[img_height, 0],  # Inverting y-axis to match image coordinates
            visible=False,
            scaleanchor="x",  # Maintains aspect ratio
            scaleratio=1,
        )
        # Ensure the container height matches the width for aspect ratio
        height = img_height * (800 / img_width) if img_width > 0 else 800
        fig.update_layout(
            height=height,
            margin=dict(l=0, r=0, t=0, b=0),
            dragmode="pan",  # This sets the hand tool as default
            showlegend=True,
            legend=dict(
                title="Lesion Types",
                orientation="v",  # Vertical legend
                yanchor="top",
                y=0.8,
                xanchor="left",
                x=0.75,  # Places legend just outside the right of the plot
                bgcolor="rgba(255, 255, 255, 0.5)",  # Semi-transparent background
            ),
            clickmode="event+select",  # Allows clicking legend to toggle visibility
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "scrollZoom": True,  # Ensure mouse wheel zooming is on
                "displayModeBar": False,  # Show the bar
                "modeBarButtonsToRemove": [
                    "select2d",
                    "lasso2d",
                ],  # Clean up unused tools
            },
        )
