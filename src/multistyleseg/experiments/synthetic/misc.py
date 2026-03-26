import streamlit as st
import torch
from torch.autograd.functional import jvp
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Jacobian Verification
def jacobian_verification(model, fine_images, coarse_images, f_index, style_direction):
    cols = st.tabs(
        [
            "Input Perturbation",
            "Linear vs Nonlinear Decomposition",
            "Quadratic Scaling",
            "Style Decomposition",
        ]
    )
    with cols[0]:
        # The input perturbation
        epsilon = fine_images - coarse_images  # [B, C, H, W]
        # Actual ΔF

        model.eval()
        with torch.no_grad():
            f_fine = model.encoder(fine_images)[f_index].mean(dim=(2, 3))  # [B, C]
            f_coarse = model.encoder(coarse_images)[f_index].mean(dim=(2, 3))
            delta_F_actual = f_fine - f_coarse  # [B, C]

        # Predicted ΔF via Jacobian: J_E @ ε
        def encoder_pooled(x):
            return model.encoder(x)[f_index].mean(dim=(2, 3))  # [B, C]

        # Use jvp to compute J_E @ ε for each image

        delta_F_predicted_list = []

        for i in range(len(coarse_images)):
            x_i = coarse_images[i : i + 1].requires_grad_(True)
            eps_i = epsilon[i : i + 1]

            # jvp computes f(x) and J @ v simultaneously
            _, delta_F_i = jvp(encoder_pooled, (x_i,), (eps_i,))
            delta_F_predicted_list.append(delta_F_i)

        delta_F_predicted = torch.cat(delta_F_predicted_list, dim=0)  # [B, C]
        st.write(f"Actual ΔF norm: {delta_F_actual.norm(dim=1).mean():.4f}")
        st.write(f"Predicted ΔF norm: {delta_F_predicted.norm(dim=1).mean():.4f}")

        cosine_sim = F.cosine_similarity(delta_F_actual, delta_F_predicted, dim=1)
        relative_error = (delta_F_actual - delta_F_predicted).norm(
            dim=1
        ) / delta_F_actual.norm(dim=1)

        st.write(
            f"Cosine similarity (actual vs predicted): {cosine_sim.mean():.4f} ± {cosine_sim.std():.4f}"
        )
        st.write(
            f"Relative error: {relative_error.mean():.4f} ± {relative_error.std():.4f}"
        )

        # Also check magnitude ratio
        magnitude_ratio = delta_F_predicted.norm(dim=1) / delta_F_actual.norm(dim=1)
        st.write(
            f"Magnitude ratio (pred/actual): {magnitude_ratio.mean():.4f} ± {magnitude_ratio.std():.4f}"
        )
        visualize_vectors(
            style_direction,
            delta_F_actual,
            delta_F_predicted,
            delta_F_actual - delta_F_predicted,
        )
    with cols[1]:
        st.subheader("Decomposing Linear vs Nonlinear Components")

        # Actual ΔF = Linear (J·ε) + Nonlinear residual
        # Let's see what the residual looks like

        delta_F_linear = delta_F_predicted  # J_E · ε
        delta_F_residual = delta_F_actual - delta_F_linear  # Nonlinear component

        st.write("Magnitude decomposition:")
        st.write(f"  ||ΔF_actual||:   {delta_F_actual.norm(dim=1).mean():.4f}")
        st.write(f"  ||ΔF_linear||:   {delta_F_linear.norm(dim=1).mean():.4f}")
        st.write(f"  ||ΔF_residual||: {delta_F_residual.norm(dim=1).mean():.4f}")

        # Check: are linear and residual orthogonal?
        cos_linear_residual = F.cosine_similarity(
            delta_F_linear, delta_F_residual, dim=1
        )
        st.write(
            f"Linear ↔ Residual cosine: {cos_linear_residual.mean():.4f} ± {cos_linear_residual.std():.4f}"
        )

        # Key question: Is the residual also aligned with the style direction?
        d_normalized = style_direction.unsqueeze(0)  # [1, C]

        cos_actual_style = F.cosine_similarity(delta_F_actual, d_normalized, dim=1)
        cos_linear_style = F.cosine_similarity(delta_F_linear, d_normalized, dim=1)
        cos_residual_style = F.cosine_similarity(delta_F_residual, d_normalized, dim=1)

        st.write("Alignment with style direction d:")
        st.write(
            f"ΔF_actual ↔ d:   {cos_actual_style.mean():.4f} ± {cos_actual_style.std():.4f}"
        )
        st.write(
            f"ΔF_linear ↔ d:   {cos_linear_style.mean():.4f} ± {cos_linear_style.std():.4f}"
        )
        st.write(
            f"ΔF_residual ↔ d: {cos_residual_style.mean():.4f} ± {cos_residual_style.std():.4f}"
        )

        # Variance explained by linear term
        variance_explained = (delta_F_linear.norm(dim=1) ** 2) / (
            delta_F_actual.norm(dim=1) ** 2
        )
        st.write(f"Variance explained by linear term: {variance_explained.mean():.4f}")
    # Second-order Taylor: f(x+ε) ≈ f(x) + J·ε + 0.5·ε^T·H·ε
    # The residual should be approximately quadratic in ε
    with cols[2]:
        st.subheader("Testing Quadratic Scaling")

        # If residual is quadratic: ||residual|| ∝ ||ε||²
        # Scale ε and check if residual scales quadratically

        scales = [0.25, 0.5, 1.0, 2.0, 4.0]
        residual_norms = []
        linear_norms = []

        for scale in scales:
            eps_scaled = epsilon * scale
            x_perturbed = coarse_images + eps_scaled

            with torch.no_grad():
                f_perturbed = model.encoder(x_perturbed)[f_index].mean(dim=(2, 3))
                f_base = model.encoder(coarse_images)[f_index].mean(dim=(2, 3))
                delta_actual = f_perturbed - f_base

            # Linear prediction scales linearly
            delta_linear = delta_F_linear * scale
            delta_residual = delta_actual - delta_linear

            residual_norms.append(delta_residual.norm(dim=1).mean().item())
            linear_norms.append(delta_linear.norm(dim=1).mean().item())

        df = pd.DataFrame(
            {
                "Scale": scales,
                "Linear_Norm": linear_norms,
                "Residual_Norm": residual_norms,
            }
        )
        df["Residual/Scale^2"] = df["Residual_Norm"] / (df["Scale"] ** 2)
        st.write("Norms at different scales:")
        st.write(df)
    with cols[3]:
        # Project everything onto 2D: style direction d and orthogonal complement
        d_norm = style_direction / style_direction.norm()

        # For each sample, decompose ΔF into style and orthogonal components
        def decompose(delta_f, d_norm):
            proj_style = (delta_f * d_norm).sum(dim=1, keepdim=True) * d_norm  # [B, C]
            proj_ortho = delta_f - proj_style
            return proj_style.norm(dim=1), proj_ortho.norm(dim=1)

        style_actual, ortho_actual = decompose(delta_F_actual[:16], d_norm)
        style_linear, ortho_linear = decompose(delta_F_linear, d_norm)
        style_residual, ortho_residual = decompose(delta_F_residual, d_norm)

        st.subheader("Decomposition into style vs orthogonal:")
        st.write(
            f"Actual:   ||style|| = {style_actual.mean():.3f}, ||ortho|| = {ortho_actual.mean():.3f}, ratio = {(style_actual / ortho_actual).mean():.2f}"
        )
        st.write(
            f"Linear:   ||style|| = {style_linear.mean():.3f}, ||ortho|| = {ortho_linear.mean():.3f}, ratio = {(style_linear / (ortho_linear + 1e-8)).mean():.2f}"
        )
        st.write(
            f"Residual: ||style|| = {style_residual.mean():.3f}, ||ortho|| = {ortho_residual.mean():.3f}, ratio = {(style_residual / (ortho_residual + 1e-8)).mean():.2f}"
        )

        # The nonlinearity INCREASES the style/ortho ratio!
        st.write("Style amplification by nonlinearity:")
        st.write(f"Linear style component:   {style_linear.mean():.3f}")
        st.write(f"Residual style component: {style_residual.mean():.3f}")
        st.write(f"Total style component:    {style_actual.mean():.3f}")
        st.write(
            f"Nonlinear boost: {style_residual.mean() / style_linear.mean():.1f}× more style from nonlinear term"
        )


def visualize_vectors(d, delta_F_actual, delta_F_linear, delta_F_residual):
    # Axis 1: Style direction (normalized)
    d_norm = (d / d.norm()).cpu()

    # Axis 2: Principal orthogonal direction (from residuals projected orthogonal to d)
    # Remove style component from residuals, then do PCA
    residuals_ortho = (
        delta_F_residual
        - (delta_F_residual @ d_norm.cuda()).unsqueeze(1) * d_norm.cuda()
    )
    U, S, V = torch.linalg.svd(residuals_ortho.cpu())
    ortho_axis = V[0]  # Top orthogonal direction

    # Ensure orthogonality
    ortho_axis = ortho_axis - (ortho_axis @ d_norm) * d_norm
    ortho_axis = ortho_axis / ortho_axis.norm()

    # Project all vectors onto 2D
    def project_2d(vectors, axis1, axis2):
        """Project [B, C] vectors onto 2D basis."""
        vectors_cpu = vectors.cpu() if vectors.is_cuda else vectors
        x = (vectors_cpu @ axis1).numpy()
        y = (vectors_cpu @ axis2).numpy()
        return x, y

    # Project each component
    actual_x, actual_y = project_2d(delta_F_actual, d_norm, ortho_axis)
    linear_x, linear_y = project_2d(delta_F_linear, d_norm, ortho_axis)
    residual_x, residual_y = project_2d(delta_F_residual, d_norm, ortho_axis)

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left plot: All samples as arrows from origin
    ax1 = axes[0]

    for i in range(min(16, len(actual_x))):
        # Linear (blue) - from origin
        ax1.arrow(
            0,
            0,
            linear_x[i],
            linear_y[i],
            head_width=0.08,
            head_length=0.05,
            fc="blue",
            ec="blue",
            alpha=0.4,
            linewidth=1.5,
        )

        # Residual (orange) - from tip of linear
        ax1.arrow(
            linear_x[i],
            linear_y[i],
            residual_x[i],
            residual_y[i],
            head_width=0.08,
            head_length=0.05,
            fc="orange",
            ec="orange",
            alpha=0.4,
            linewidth=1.5,
        )

        # Actual (green) - from origin (should equal linear + residual)
        ax1.arrow(
            0,
            0,
            actual_x[i],
            actual_y[i],
            head_width=0.08,
            head_length=0.05,
            fc="green",
            ec="green",
            alpha=0.3,
            linewidth=1,
        )

    # Style direction reference (thick arrow along x-axis)
    max_range = max(np.abs(actual_x).max(), np.abs(actual_y).max()) * 1.2
    ax1.arrow(
        0,
        0,
        max_range * 0.9,
        0,
        head_width=0.1,
        head_length=0.08,
        fc="red",
        ec="red",
        linewidth=3,
        label="Style direction",
    )

    # Legend with proxy artists

    ax1.plot([], [], color="blue", linewidth=2, label="Linear (J·ε)")
    ax1.plot([], [], color="orange", linewidth=2, label="Nonlinear (residual)")
    ax1.plot([], [], color="green", linewidth=2, label="Total (actual)")
    ax1.plot([], [], color="red", linewidth=3, label="Style direction")

    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    ax1.axvline(x=0, color="gray", linestyle="--", alpha=0.3)
    ax1.set_xlim(-max_range * 0.3, max_range)
    ax1.set_ylim(-max_range * 0.5, max_range * 0.5)
    ax1.set_xlabel("Style axis", fontsize=12)
    ax1.set_ylabel("Orthogonal axis", fontsize=12)
    ax1.set_title(
        "Per-Sample Feature Changes\n(Linear + Nonlinear = Actual)", fontsize=12
    )
    ax1.legend(loc="upper left")
    ax1.set_aspect("equal")

    # Right plot: Mean vectors (clearer summary)
    ax2 = axes[1]

    mean_linear = np.array([linear_x.mean(), linear_y.mean()])
    mean_residual = np.array([residual_x.mean(), residual_y.mean()])
    mean_actual = np.array([actual_x.mean(), actual_y.mean()])

    # Draw mean vectors
    ax2.arrow(
        0,
        0,
        mean_linear[0],
        mean_linear[1],
        head_width=0.12,
        head_length=0.08,
        fc="blue",
        ec="blue",
        linewidth=3,
        label=f"Linear (J·ε)",
    )
    ax2.arrow(
        mean_linear[0],
        mean_linear[1],
        mean_residual[0],
        mean_residual[1],
        head_width=0.12,
        head_length=0.08,
        fc="orange",
        ec="orange",
        linewidth=3,
        label="Nonlinear (residual)",
    )
    ax2.arrow(
        0,
        0,
        mean_actual[0],
        mean_actual[1],
        head_width=0.12,
        head_length=0.08,
        fc="green",
        ec="green",
        linewidth=1,
        label="Total (actual)",
    )

    # Style direction
    ax2.arrow(
        0,
        0,
        max_range * 0.9,
        0,
        head_width=0.1,
        head_length=0.08,
        fc="red",
        ec="red",
        linewidth=2,
        alpha=0.5,
    )

    # Annotations
    ax2.annotate(
        "Linear",
        xy=(mean_linear[0] / 2, mean_linear[1] / 2 - 0.3),
        fontsize=10,
        color="blue",
    )
    ax2.annotate(
        "Residual",
        xy=(
            mean_linear[0] + mean_residual[0] / 2,
            mean_linear[1] + mean_residual[1] / 2 + 0.4,
        ),
        fontsize=10,
        color="orange",
    )
    ax2.annotate(
        "Total",
        xy=(mean_actual[0] / 2 + 0.3, mean_actual[1] / 2 + 0.2),
        fontsize=10,
        color="green",
    )

    # Show angle to style axis
    angle_linear = np.arctan2(mean_linear[1], mean_linear[0]) * 180 / np.pi
    angle_actual = np.arctan2(mean_actual[1], mean_actual[0]) * 180 / np.pi

    ax2.set_title(
        f"Mean Feature Change Vectors\n"
        f"Linear angle: {angle_linear:.1f}°, Actual angle: {angle_actual:.1f}°",
        fontsize=12,
    )

    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    ax2.axvline(x=0, color="gray", linestyle="--", alpha=0.3)
    ax2.set_xlim(-1, max_range)
    ax2.set_ylim(-max_range * 0.4, max_range * 0.4)
    ax2.set_xlabel("Style axis", fontsize=12)
    ax2.set_ylabel("Orthogonal axis", fontsize=12)
    ax2.legend(loc="upper left")
    ax2.set_aspect("equal")

    plt.tight_layout()
    st.pyplot(fig)
