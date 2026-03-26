import cv2
import numpy as np
from enum import Enum
from torch.utils.data import IterableDataset
import torch
import matplotlib.pyplot as plt


class Style(Enum):
    FINE = 0
    COARSE = 1
    COARSER = 2


class COLORS:
    FINE = (255, 0, 0)   
    COARSER = (245, 0, 0)
    COARSE = (235, 0, 0) 
    

MAPPING = {
    Style.FINE: COLORS.FINE,
    Style.COARSE: COLORS.COARSE,
    Style.COARSER: COLORS.COARSER
}

def get_circumcircle(pts):
    """
    Get the circumcircle of a triangle defined by pts (3,2)
    Returns center (x,y) and radius
    """
    A = pts[0]
    B = pts[1]
    C = pts[2]
    
    D = 2 * (A[0] * (B[1] - C[1]) + B[0] * (C[1] - A[1]) + C[0] * (A[1] - B[1]))
    if D == 0:
        return (0, 0), 0  # Degenerate case
    
    Ux = ((A[0]**2 + A[1]**2) * (B[1] - C[1]) + (B[0]**2 + B[1]**2) * (C[1] - A[1]) + (C[0]**2 + C[1]**2) * (A[1] - B[1])) / D
    Uy = ((A[0]**2 + A[1]**2) * (C[0] - B[0]) + (B[0]**2 + B[1]**2) * (A[0] - C[0]) + (C[0]**2 + C[1]**2) * (B[0] - A[0])) / D
    
    center = (int(Ux), int(Uy))
    radius = int(np.linalg.norm(np.array(center) - A))
    
    return center, radius

def get_bbox(pts):
    """
    Get the bounding box of a triangle defined by pts (3,2)
    Returns top-left (x,y), bottom-right (x,y)
    """
    x_min = int(np.min(pts[:, 0]))
    x_max = int(np.max(pts[:, 0]))
    y_min = int(np.min(pts[:, 1]))
    y_max = int(np.max(pts[:, 1]))
    
    top_left = (x_min, y_min)
    bottom_right = (x_max, y_max)
    
    return top_left, bottom_right

def get_incircle(pts):
    """
    Get the incircle of a triangle defined by pts (3,2)
    Returns center (x,y) and radius
    """
    A = pts[0]
    B = pts[1]
    C = pts[2]
    
    a = np.linalg.norm(B - C)
    b = np.linalg.norm(A - C)
    c = np.linalg.norm(A - B)
    
    perimeter = a + b + c
    if perimeter == 0:
        return (0, 0), 0  # Degenerate case
    
    Ix = (a * A[0] + b * B[0] + c * C[0]) / perimeter
    Iy = (a * A[1] + b * B[1] + c * C[1]) / perimeter
    
    s = perimeter / 2
    area = np.sqrt(s * (s - a) * (s - b) * (s - c))
    radius = int(area / s)
    
    center = (int(Ix), int(Iy))
    
    return center, radius
    
    

def synthetize_data(style, resolution, n_shapes):
    """
    Synthetize an image with random shapes and a corresponding mask.
    The shapes are colors triangles. The corresponding mask is either the bbox, the circumcircle, the exact triangle.
    """
    
    image = np.zeros((resolution, resolution, 3), dtype=np.uint8) 
    mask = np.zeros((resolution, resolution), dtype=np.uint8)
    # Random background color
    bg_color = np.random.randint(0, 256, size=3)
    image[:] = bg_color
    shape_color = MAPPING[style]
    for _ in range(n_shapes):
        
        # Sample a random triangle, ensure its not too flat
        while True:
            pts = np.random.randint(0, resolution, size=(3, 2))
            a = np.linalg.norm(pts[0] - pts[1])
            b = np.linalg.norm(pts[1] - pts[2])
            c = np.linalg.norm(pts[2] - pts[0])
            s = (a + b + c) / 2
            area = np.sqrt(s * (s - a) * (s - b) * (s - c))
            if area > 0.01 * (resolution ** 2):
                break
        
        
        # We want 
        cv2.fillPoly(image, [pts], shape_color)
        
        if style == Style.FINE:
            cv2.fillPoly(mask, [pts], 255)
        elif style == Style.COARSE:
            center, radius = get_incircle(pts)
            cv2.circle(mask, center, radius, 255, -1)
        elif style == Style.COARSER:
            top_left, bottom_right = get_bbox(pts)
            cv2.rectangle(mask, top_left, bottom_right, 255, -1)
    return image, mask
    

class TriangleDataset(IterableDataset):
    def __init__(self, style, resolution=128, n_shapes=5):
        if not isinstance(style, list):
            style = [style]
        self.style = style
        self.resolution = resolution
        self.n_shapes = n_shapes
        
    def __getitem__(self, idx):
        # Randomly choose a style for each item
        current_style = np.random.choice(self.style)
        image, mask = synthetize_data(current_style, self.resolution, self.n_shapes)
        image = image.transpose(2, 0, 1) / 255.0  # Normalize to [0,1] and change to C,H,W
        mask = mask[np.newaxis, :, :] / 255.0      # Add channel dimension and normalize to [0,1]
        # Returns tensors
        
        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32), current_style.value
    
    def __iter__(self):
        idx = 0
        while True:
            yield self.__getitem__(idx)
            idx += 1
    
    def plot(self):
        image, mask, style = self.__getitem__(0)
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        
        
        axs[0].imshow(image.permute(1, 2, 0).numpy())
        axs[0].set_title("Image")
        axs[0].axis('off')
        axs[1].imshow(mask.squeeze().numpy(), cmap='gray')
        axs[1].set_title(f"Mask ({Style(style).name})")
        axs[1].axis('off')
        plt.show()
        
        


def swap_colors_tensors(images, from_color, to_color):
    """
    Swap from_color to to_color in a batch of images (tensors).
    images: (B, C, H, W)
    from_color, to_color: (3,) tuples
    """
    from_color = torch.tensor(from_color, dtype=images.dtype, device=images.device).view(1, 3, 1, 1) / 255.0
    to_color = torch.tensor(to_color, dtype=images.dtype, device=images.device).view(1, 3, 1, 1) / 255.0
    
    mask = torch.all(images == from_color, dim=1, keepdim=True)  # (B, 1, H, W)
    images = images * (~mask) + to_color * mask
    return images

def swap_expected_style_tensors(images, current_style, target_style):
    """
    Swap the colors in images tensor from current_style to target_style.
    images: (B, C, H, W)
    current_style, target_style: Style enums
    """
    from_color = MAPPING[Style(current_style)]
    to_color = MAPPING[Style(target_style)]
    return swap_colors_tensors(images, from_color, to_color)
    
def get_dataloader(style, resolution=128, n_shapes=5, batch_size=16, n_workers=0):
    dataset = TriangleDataset(style, resolution, n_shapes)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=n_workers, pin_memory=True)
    return dataloader