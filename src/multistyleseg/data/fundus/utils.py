from multistyleseg.data.fundus.factory import FundusDataset
import torch
from typing import List

MAPPING = {
    FundusDataset.IDRID: 0,
    FundusDataset.FGADR: 1,
    FundusDataset.MESSIDOR: 2,
    FundusDataset.DDR: 3,
    FundusDataset.RETLES: 4,
}


def convert_list_datasets_to_tensor(
    list_datasets: List[FundusDataset],
    device=None,
) -> torch.Tensor:
    """Convert a list of FundusDataset enums to a tensor of their integer values.

    Args:
        list_datasets (List[FundusDataset]): List of FundusDataset enums.

    Returns:
        torch.Tensor: Tensor containing the integer values of the datasets.
    """
    dataset_values = [MAPPING[dataset] for dataset in list_datasets]
    if device:
        return torch.tensor(dataset_values, dtype=torch.long, device=device)
    return torch.tensor(dataset_values, dtype=torch.long)
