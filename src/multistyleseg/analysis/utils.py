import pandas as pd
import numpy as np

DATASET_COLORS = ["#ca9ee6", "#e78284", "#ef9f76", "#81c8be", "#8caaee"]

LESION_ORDER = ["COTTON_WOOL_SPOT", "EXUDATES", "HEMORRHAGES", "MICROANEURYSMS"]


class LesionsUtils:
    COTTON_WOOL_SPOT = "#4b8a9f"
    EXUDATES = "#14597e"
    HEMORRHAGES = "#5b2994"
    MICROANEURYSMS = "#933d93"

    @staticmethod
    def get_color(lesion_name):
        lesion_name = LesionsUtils._closest_match(lesion_name)
        mapping = {
            "COTTON_WOOL_SPOT": LesionsUtils.COTTON_WOOL_SPOT,
            "EXUDATES": LesionsUtils.EXUDATES,
            "HEMORRHAGES": LesionsUtils.HEMORRHAGES,
            "MICROANEURYSMS": LesionsUtils.MICROANEURYSMS,
        }
        return mapping.get(lesion_name, "#000000")  # Default to black if not found

    @staticmethod
    def get_rgb(lesion_name):
        color_hex = LesionsUtils.get_color(lesion_name)
        return np.asarray([int(color_hex[i : i + 2], 16) for i in (1, 3, 5)]) / 255.0

    @staticmethod
    def _closest_match(lesion_name):
        lesions_name = lesion_name.upper()
        if ("COTTON" in lesions_name) or ("CWS" in lesions_name):
            return "COTTON_WOOL_SPOT"
        elif ("EXUDATES" in lesions_name) or ("EX" in lesions_name):
            return "EXUDATES"
        elif "HE" in lesions_name:
            return "HEMORRHAGES"
        elif ("MICRO" in lesions_name) or ("μ" in lesions_name) or ("MA" in lesions_name):
            return "MICROANEURYSMS"

    @staticmethod
    def reorder(lesions: list[str] | pd.Series):
        """
        Reorder the input lesions inplace according to the order defined in the MAPPING_STR dictionary.
        This ensures that the lesions are always in the same order in the plots, regardless of the order they appear in the input data.
        We also make sure the name passed matches the keys in the MAPPING_STR dictionary, so that we can use the mapping to get the correct color for each lesion.
        We do not modify the items in the list, only reorder them according to the defined order.
        """

        lesions = pd.Series(lesions)
        lesion_to_index = {lesion: idx for idx, lesion in enumerate(LESION_ORDER)}
        renamed_lesions = lesions.apply(LesionsUtils._closest_match)
        sorted_indices = renamed_lesions.map(lesion_to_index)
        return lesions.iloc[sorted_indices.argsort()]

    @staticmethod
    def sort_key(lesion_name: str) -> int:
        """Return the canonical sort index for a lesion type."""
        matched = LesionsUtils._closest_match(lesion_name)
        try:
            return LESION_ORDER.index(matched)
        except ValueError:
            return len(LESION_ORDER)  # unknown → end

    @staticmethod
    def sort_key_from_series(lesion_series: pd.Series) -> pd.Series:
        """Return a Series of sort keys for a Series of lesion names."""
        return lesion_series.apply(LesionsUtils.sort_key)
