import math
from enum import Enum
from typing import Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader


class AnnotationType(Enum):
    FINE: int = 0
    COARSE: int = 1


class Color(Enum):
    FINE = (255, 0, 0)  # Red
    COARSE = (245, 0, 0)


class Task(Enum):
    COLOR_BASED = 0
    TEXTURE_BASED = 1


class Angle(Enum):
    FINE: int = 90
    COARSE: int = 85


def color_mapping(annotation_type: AnnotationType) -> Color:
    match annotation_type:
        case AnnotationType.FINE:
            return Color.FINE
        case AnnotationType.COARSE:
            return Color.COARSE
        case _:
            raise ValueError(f"Unknown annotation type: {annotation_type}")


def angle_mapping(annotation_type: AnnotationType) -> Angle:
    match annotation_type:
        case AnnotationType.FINE:
            return Angle.FINE
        case AnnotationType.COARSE:
            return Angle.COARSE
        case _:
            raise ValueError(f"Unknown annotation type: {annotation_type}")


def obtuse(x1, y1, x2, y2, x3, y3):
    sideAB = sideLength(x1, y1, x2, y2)
    sideBC = sideLength(x2, y2, x3, y3)
    sideAC = sideLength(x3, y3, x1, y1)
    largest = max(sideAB, sideBC, sideAC)
    var1 = min(sideAB, sideBC, sideAC)
    if sideAB != largest and sideAB != var1:
        var2 = sideAB
    elif sideBC != largest and sideBC != var1:
        var2 = sideBC
    else:
        var2 = sideAC
    if (largest) ** 2 > ((var1) ** 2 + (var2) ** 2):
        return True
    else:
        return False


def sideLength(x1, y1, x2, y2):
    length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return length


def find_circumcircle(p1, p2, p3):
    """
    Calculates the circumcenter and circumradius of a triangle defined by three points.

    Args:
        p1, p2, p3 (tuple or list): Coordinates of the three vertices (x, y).

    Returns:
        tuple: (circumcenter_x, circumcenter_y, circumradius)
               Returns None if the points are collinear (degenerate triangle).
    """
    A = np.array(p1)
    B = np.array(p2)
    C = np.array(p3)

    # Vectors representing the sides
    AB = B - A
    AC = C - A

    # Check for collinearity
    if np.cross(AB, AC) == 0:
        return None  # Points are collinear

    # Perpendicular bisector equations
    # For a line segment (x1, y1) to (x2, y2), the perpendicular bisector
    # passes through the midpoint ((x1+x2)/2, (y1+y2)/2) and has a slope
    # of -1/m, where m is the slope of the segment.
    # The equation of a line is Ax + By = C

    # Solve the system of linear equations to find the circumcenter
    # A_matrix = [[n_AB[0], n_AB[1]], [n_BC[0], n_BC[1]]]
    # B_vector = [c_AB, c_BC]
    # circumcenter = np.linalg.solve(A_matrix, B_vector)

    # Using a more direct formula for circumcenter calculation based on determinants
    D = 2 * (A[0] * (B[1] - C[1]) + B[0] * (C[1] - A[1]) + C[0] * (A[1] - B[1]))

    if D == 0:  # Degenerate triangle, points are collinear
        return None

    Ux = (
        (A[0] ** 2 + A[1] ** 2) * (B[1] - C[1])
        + (B[0] ** 2 + B[1] ** 2) * (C[1] - A[1])
        + (C[0] ** 2 + C[1] ** 2) * (A[1] - B[1])
    ) / D
    Uy = (
        (A[0] ** 2 + A[1] ** 2) * (C[0] - B[0])
        + (B[0] ** 2 + B[1] ** 2) * (A[0] - C[0])
        + (C[0] ** 2 + C[1] ** 2) * (B[0] - A[0])
    ) / D

    circumcenter = np.array([Ux, Uy])

    # Calculate circumradius
    circumradius = np.linalg.norm(circumcenter - A)

    return circumcenter[0], circumcenter[1], circumradius


def find_bounding_box(points: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Find the bounding box of a set of points.

    Args:
        points (np.ndarray): Array of points with shape (N, 2).

    Returns:
        Tuple[int, int, int, int]: (min_x, min_y, max_x, max_y)
    """
    min_x = int(np.min(points[:, 0]))
    min_y = int(np.min(points[:, 1]))
    max_x = int(np.max(points[:, 0]))
    max_y = int(np.max(points[:, 1]))
    return min_x, min_y, max_x, max_y


def synthesize_image(
    resolution: int,
    expected_style: AnnotationType,
    n_shapes: int,
    task: Task = Task.COLOR_BASED,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Draw random filled triangles on a blank image and return the image and the corresponding masks (coarse and fine).
    """
    random_color = np.random.randint(0, 255, (1, 1, 3), dtype=np.uint8)
    image = np.ones((resolution, resolution, 3), dtype=np.uint8) * random_color
    fine_mask = np.zeros((resolution, resolution), dtype=np.uint8)
    coarse_mask = np.zeros((resolution, resolution), dtype=np.uint8)
    if task == Task.COLOR_BASED:
        color = color_mapping(expected_style).value
    else:
        xx, yy = np.meshgrid(np.arange(resolution), np.arange(resolution))
        angle = angle_mapping(expected_style).value
        pattern = (
            np.sin(xx * np.cos(np.deg2rad(angle)) + yy * np.sin(np.deg2rad(angle))) + 1
        ) / 2
        pattern = (pattern * 255).astype(np.uint8)

    for _ in range(n_shapes):
        # Sample a random triangle, ensure its not too flat
        while True:
            pts = np.random.randint(0, resolution, size=(3, 2))
            a = np.linalg.norm(pts[0] - pts[1])
            b = np.linalg.norm(pts[1] - pts[2])
            c = np.linalg.norm(pts[2] - pts[0])
            s = (a + b + c) / 2
            area = np.sqrt(s * (s - a) * (s - b) * (s - c))
            if area > 0.001 * (resolution**2):
                break
        if task == Task.COLOR_BASED:
            cv2.fillPoly(image, [pts], color)
        else:
            mask = np.zeros((resolution, resolution), dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 1)
            for c in range(3):
                image[:, :, c] = np.where(mask == 1, pattern, image[:, :, c])
        cv2.fillPoly(fine_mask, [pts], 1)

        min_x, min_y, max_x, max_y = find_bounding_box(pts)
        cv2.rectangle(coarse_mask, (min_x, min_y), (max_x, max_y), 1, thickness=-1)

    image = (image.transpose(2, 0, 1) / 255.0).astype(
        np.float32
    )  # Convert to CxHxW format
    data = {"image": image, "fine_mask": fine_mask, "coarse_mask": coarse_mask}
    return data


def replace_color_tensor(
    images: torch.Tensor,
    from_color: Color | Tuple[int, int, int],
    to_color: Color | Tuple[int, int, int],
) -> torch.Tensor:
    """
    Replace one color with another in a batch of images (unidirectional).
    """
    if isinstance(from_color, Color):
        from_color = (
            torch.tensor(from_color.value, device=images.device).view(1, 3, 1, 1)
            / 255.0
        )
    else:
        from_color = (
            torch.tensor(from_color, device=images.device).view(1, 3, 1, 1) / 255.0
        )

    if isinstance(to_color, Color):
        to_color = (
            torch.tensor(to_color.value, device=images.device).view(1, 3, 1, 1) / 255.0
        )
    else:
        to_color = torch.tensor(to_color, device=images.device).view(1, 3, 1, 1) / 255.0

    tol = 0
    mask = torch.all(
        torch.isclose(images, from_color, atol=tol, rtol=0.0), dim=1, keepdim=True
    )

    # Only replace from_color → to_color, leave everything else unchanged
    result = images * (~mask) + to_color * mask
    return result


def swap_annotators_tensor(
    data: dict,
    from_annotator: AnnotationType,
    to_annotator: AnnotationType,
):
    """
    Swap the annotator colors in the images.

    Args:
        images (torch.Tensor): Tensor of images with shape (N, C, H, W) or (C, H, W).
            If a 3D tensor is provided, it will be treated as a batch of one.

    Returns:
        torch.Tensor: Tensor of images with annotator colors swapped.
    """
    images = data["image"]
    task = data["task"][0].item()
    task = Task(task)
    if task == Task.COLOR_BASED:
        if images.ndim == 3:
            images = images.unsqueeze(0)
        elif images.ndim != 4:
            raise ValueError(
                f"Expected images of shape (N, C, H, W) or (C, H, W), got shape {images.shape}"
            )
        results = images.clone()
        color1 = color_mapping(from_annotator)
        color2 = color_mapping(to_annotator)

        results = replace_color_tensor(results, color1, color2)
    else:
        if images.ndim == 3:
            images = images.unsqueeze(0)
        elif images.ndim != 4:
            raise ValueError(
                f"Expected images of shape (N, C, H, W) or (C, H, W), got shape {images.shape}"
            )
        results = images.clone()
        angle1 = angle_mapping(from_annotator).value
        angle2 = angle_mapping(to_annotator).value
        N, C, H, W = results.shape
        xx, yy = torch.meshgrid(
            torch.arange(W, device=results.device),
            torch.arange(H, device=results.device),
            indexing="xy",
        )
        pattern1 = (
            torch.sin(
                xx * math.cos(math.radians(angle1))
                + yy * math.sin(math.radians(angle1))
            )
            + 1
        ) / 2
        pattern2 = (
            torch.sin(
                xx * math.cos(math.radians(angle2))
                + yy * math.sin(math.radians(angle2))
            )
            + 1
        ) / 2
        pattern1 = pattern1.unsqueeze(0).unsqueeze(0)  # Shape (1, 1, H, W)
        pattern2 = pattern2.unsqueeze(0).unsqueeze(0)  # Shape (1, 1, H, W)

        # Create a mask where pattern1 is present
        tol = 0.01
        mask = torch.all(
            torch.isclose(results, pattern1, atol=tol, rtol=0.0), dim=1, keepdim=True
        )

        # Replace pattern1 with pattern2 using the mask
        results = results * (~mask) + pattern2 * mask

    return results
