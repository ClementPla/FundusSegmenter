from tqdm.auto import tqdm
import torch
from torchmetrics import Accuracy
from torchmetrics.segmentation import MeanIoU
import torch.nn as nn
import streamlit as st
from multistyleseg.data.synthetic import SynthTriangle, AnnotationType
from multistyleseg.data.synthetic.utils import Task
from multistyleseg.utils import FOLDER_SYNTHETIC
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from monai.losses.dice import DiceCELoss
from multistyleseg.experiments.synthetic.utils import get_dataset
import torch.nn.functional as F

RETRAIN = False


def train_step(images, masks, model, optimizer, scheduler, metrics, loss_fn):
    masks = masks.unsqueeze(1).float()  # Add channel dimension and convert to float

    model.train()
    images = images.to("cuda")
    masks = masks.to("cuda")

    optimizer.zero_grad()
    outputs = model(images)

    loss = loss_fn(outputs, masks)
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss.item(), metrics((torch.sigmoid(outputs) > 0.5).long(), masks.long())


def train_synthetic_model(
    model, dataloader, criterion, optimizer, scheduler, iterations
):
    metrics = MeanIoU(num_classes=2, input_format="index").cuda()
    running_loss = 0.0
    j = 0
    progress_bar = st.progress(0, text="Training Progress")

    for batch in tqdm(dataloader, total=iterations):
        image = batch["image"]
        fine_mask = batch["fine_mask"]
        coarse_mask = batch["coarse_mask"]
        annotation_types = batch["expected_style"].view(-1, 1, 1)
        mask = fine_mask * (1 - annotation_types) + coarse_mask * annotation_types
        # Get a batch of data
        image = image.to("cuda")  # Normalize to [0, 1]
        mask = mask.to("cuda")
        metrics.reset()
        loss, metric_values = train_step(
            image, mask, model, optimizer, scheduler, metrics, criterion
        )
        running_loss += loss
        if j % 50 == 0:
            print(f"Iteration {j}, Loss: {loss:.4f}, Metrics: {metric_values}")
        progress_bar.progress(
            j / iterations,
            text=f"Training Progress: Iteration {j}/{iterations}, Metrics: {metric_values}",
        )
        j += 1
        if j > iterations:
            break
    progress_bar.empty()
    return model


def train_linear_probe(model, dataloader, f_index, iterations):
    out_channels = model.encoder.out_channels[f_index]
    criterion = nn.BCEWithLogitsLoss()
    metric = Accuracy(task="binary", average="macro").cuda()

    linear_probe = nn.Sequential(
        nn.Conv2d(out_channels, 64, kernel_size=3),
        nn.ReLU(),
        nn.Conv2d(64, 32, kernel_size=3),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(32, 1),
    )
    optimizer = torch.optim.Adam(linear_probe.parameters(), lr=1e-3)
    linear_probe = linear_probe.cuda()
    model = model.cuda()
    model.eval()
    progress_bar = st.progress(0, text="Linear Probe Training Progress")
    for i, batch in tqdm(enumerate(dataloader), total=iterations):
        images = batch["image"]
        styles = batch["expected_style"].view(-1)
        images = images.cuda()
        styles = styles.cuda()
        with torch.no_grad():
            features = model.encoder(images)[f_index]
            pooled_features = features  # Global average pooling

        outputs = linear_probe(pooled_features).squeeze(1)
        loss = criterion(outputs, styles.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = (torch.sigmoid(outputs) > 0.5).float()
        metric.update(preds.cpu(), styles.cpu())
        acc = metric.compute().item()

        progress_bar.progress(
            (i + 1) / iterations,
            text=f"Linear Probe Training Progress: Iteration {i + 1}/{iterations}, Accuracy: {acc:.4f}",
        )
        if (i + 1) % 10 == 0:
            print(
                f"Linear Probe Training Iteration {i + 1}, Loss: {loss.item():.4f}, Accuracy: {acc:.4f}"
            )
            metric.reset()

        if i + 1 >= iterations:
            break
    progress_bar.empty()

    return linear_probe


@st.cache_resource(show_spinner=False)
def get_probe_linear_model(
    f_index: int, n_iterations: int = 500, task: Task = Task.COLOR_BASED
):
    if (
        FOLDER_SYNTHETIC / f"synthetic_linear_probe_findex_{f_index}_{task.value}.pth"
    ).exists() and not RETRAIN:
        model = get_joint_model(task=task)
        out_channels = model.encoder.out_channels[f_index]
        linear_probe = nn.Sequential(
            nn.Conv2d(out_channels, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, 1),
        )
        linear_probe = linear_probe.cuda()
        linear_probe.load_state_dict(
            torch.load(
                FOLDER_SYNTHETIC
                / f"synthetic_linear_probe_findex_{f_index}_{task.value}.pth"
            )
        )
        linear_probe.eval()
        return linear_probe
    dataset = get_dataset(
        n_shapes=1,
        return_all_styles=False,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=64,
        num_workers=8,
        pin_memory=True,
    )
    model = get_joint_model(task=task)
    linear_probe = train_linear_probe(
        model,
        dataloader,
        f_index=f_index,
        iterations=n_iterations,
    )
    torch.save(
        linear_probe.state_dict(),
        FOLDER_SYNTHETIC / f"synthetic_linear_probe_findex_{f_index}_{task.value}.pth",
    )
    return linear_probe


@st.cache_resource(show_spinner=False)
def get_joint_model(task: Task = Task.COLOR_BASED):
    if (
        FOLDER_SYNTHETIC / f"synthetic_joint_model_{task.value}.pth"
    ).exists() and not RETRAIN:
        model = smp.create_model("unet", in_channels=3, out_channels=1)
        model = model.cuda()
        model.load_state_dict(
            torch.load(FOLDER_SYNTHETIC / f"synthetic_joint_model_{task.value}.pth")
        )
        model.eval()
        return model
    dataset = get_dataset(
        return_all_styles=False,
    )
    model = smp.create_model("unet", in_channels=3, out_channels=1)
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        num_workers=8,
        pin_memory=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
    model.train()
    model = model.cuda()
    model = train_synthetic_model(
        model,
        dataloader,
        criterion=DiceCELoss(
            sigmoid=True,
            label_smoothing=0.3,
            include_background=True,
        ),
        optimizer=optimizer,
        scheduler=scheduler,
        iterations=1000,
    )
    torch.save(
        model.state_dict(), FOLDER_SYNTHETIC / f"synthetic_joint_model_{task.value}.pth"
    )
    return model


@st.cache_resource(show_spinner=False)
def get_sequential_model(task: Task = Task.COLOR_BASED):
    model1 = model = None
    if (
        FOLDER_SYNTHETIC / f"sequential_first_sequential_model_{task.value}.pth"
    ).exists() and not RETRAIN:
        model1 = smp.create_model("unet", in_channels=3, out_channels=1)
        model1 = model1.cuda()
        model1.load_state_dict(
            torch.load(
                FOLDER_SYNTHETIC / f"sequential_first_sequential_model_{task.value}.pth"
            )
        )
        model1.eval()
    if (
        FOLDER_SYNTHETIC / f"sequential_complete_model_{task.value}.pth"
    ).exists() and not RETRAIN:
        model = smp.create_model("unet", in_channels=3, out_channels=1)
        model = model.cuda()
        model.load_state_dict(
            torch.load(FOLDER_SYNTHETIC / f"sequential_complete_model_{task.value}.pth")
        )
        model.eval()
    if model1 is not None and model is not None:
        return model1, model
    model = smp.create_model("unet", in_channels=3, out_channels=1)
    dataset = get_dataset(return_all_styles=False, annotation_type=AnnotationType.FINE)
    dataloader = DataLoader(
        dataset,
        batch_size=64,
        num_workers=8,
        pin_memory=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
    model.train()
    model = model.cuda()
    model = train_synthetic_model(
        model,
        dataloader,
        criterion=DiceCELoss(
            sigmoid=True,
            label_smoothing=0.3,
            include_background=True,
        ),
        optimizer=optimizer,
        scheduler=scheduler,
        iterations=1000,
    )
    torch.save(
        model.state_dict(),
        FOLDER_SYNTHETIC / f"sequential_first_sequential_model_{task.value}.pth",
    )
    dataset = get_dataset(
        return_all_styles=False, annotation_type=AnnotationType.COARSE
    )

    dataloader = DataLoader(
        dataset,
        batch_size=64,
        num_workers=8,
        pin_memory=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
    model.train()
    model = train_synthetic_model(
        model,
        dataloader,
        criterion=DiceCELoss(
            sigmoid=True,
            label_smoothing=0.3,
            include_background=True,
        ),
        optimizer=optimizer,
        scheduler=scheduler,
        iterations=1000,
    )

    torch.save(
        model.state_dict(),
        FOLDER_SYNTHETIC / f"sequential_complete_model_{task.value}.pth",
    )

    model1 = smp.create_model("unet", in_channels=3, out_channels=1)
    model1 = model1.cuda()
    model1.load_state_dict(
        torch.load(
            FOLDER_SYNTHETIC / f"sequential_first_sequential_model_{task.value}.pth"
        )
    )
    model1.eval()
    return model1, model


def adversarial_steer(
    model, probe, f_index, images, target_style, epsilon=0.1, alpha=0.01, n_iters=20
):
    images_adv = images.clone().requires_grad_(True)
    original = images.clone()
    target = torch.full((len(images),), target_style, device=images.device).float()

    for i in range(n_iters):
        features = model.encoder(images_adv)[f_index]
        logits = probe(features).squeeze()
        loss = F.binary_cross_entropy_with_logits(logits, target)
        loss.backward()

        with torch.no_grad():
            images_adv = images_adv - alpha * images_adv.grad.sign()
            images_adv = torch.clamp(images_adv, original - epsilon, original + epsilon)
            images_adv = torch.clamp(images_adv, 0, 1)

        images_adv = images_adv.clone().requires_grad_(True)

    return images_adv.detach()
