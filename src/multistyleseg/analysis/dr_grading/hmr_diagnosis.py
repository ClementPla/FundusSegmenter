import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# 1. STANDARDIZE RETINOPATHY (Rétinopathie diabétique)
# =============================================================================


def standardize_retinopathy(value):
    """Map raw retinopathy values to standardized categories."""
    if pd.isna(value):
        return "Ungradable"

    value = str(value).strip()

    mapping = {
        # No DR
        "Absente": "R0 - No DR",
        "R0": "R0 - No DR",
        # Mild NPDR
        "Rétinopathie non proliférante très légère": "R1 - Mild NPDR",
        "Rétinopathie non proliférante légère": "R1 - Mild NPDR",
        "R1": "R1 - Mild NPDR",
        # Moderate NPDR
        "Rétinopathie non proliférante modérée": "R2 - Moderate NPDR",
        "R2": "R2 - Moderate NPDR",
        # Severe NPDR
        "Rétinopathie non proliférante grave": "R3 - Severe NPDR",
        # Generic NPDR (unclear severity) - conservative: moderate
        "Rétinopathie non proliférante": "R2 - Moderate NPDR",
        # Possible/Suspected DR
        "Rétinopathie non proliférante possible": "R1 - Suspected NPDR",
        "Rétinopathie proliférante possible": "R4 - Suspected PDR",
        # Treated with laser (photocoagulation scars)
        "Cicatrices de photocoagulation de laser avec rétinopathie non proliférante active": "R6 - Treated, active NPDR",
        "Cicatrice de photocoagulation de laser avec rétinopathie non proliférante inactive": "R6 - Treated, inactive NPDR",
        "Cicatrice de photocoagulation de laser avec rétinopathie proliférante inactive": "R6 - Treated, inactive PDR",
        "R6": "R6 - Treated",
        # Ungradable
        "Qualité de la photographie insuffisante": "Ungradable",
        "Non applicable": "Ungradable",
    }

    return mapping.get(value, "Ungradable")


def get_retinopathy_severity(standardized_value):
    """Return numeric severity for sorting (0-6, 99 for ungradable)."""
    severity_map = {
        "R0 - No DR": 0,
        "R1 - Mild NPDR": 1,
        "R1 - Suspected NPDR": 1,
        "R2 - Moderate NPDR": 2,
        "R3 - Severe NPDR": 3,
        "R4 - Suspected PDR": 4,
        "R6 - Treated, active NPDR": 5,
        "R6 - Treated, inactive NPDR": 5,
        "R6 - Treated, inactive PDR": 5,
        "R6 - Treated": 5,
        "Ungradable": 99,
    }
    return severity_map.get(standardized_value, 99)


# =============================================================================
# 2. STANDARDIZE MACULAR THREAT (Menace diabétique de la macula)
# =============================================================================


def standardize_macular_threat(value):
    """Map raw macular threat values to standardized categories."""
    if pd.isna(value):
        return "Ungradable"

    value = str(value).strip()

    mapping = {
        # No macular threat
        "Absente": "M0 - No DME",
        "M0": "M0 - No DME",
        # Macular threat present - peripheral (2 disc diameters from fovea)
        "Présente à 2 dd de la fovéa": "M1 - DME >1dd from fovea",
        "M1": "M1 - DME >1dd from fovea",
        # Macular threat present - central (1 disc diameter from fovea)
        "Présente à 1 dd de la fovéa": "M2 - DME ≤1dd from fovea",
        "M2/OCT-/E-": "M2 - DME ≤1dd (OCT-/E-)",
        # Treated/stable maculopathy
        "M6": "M6 - Treated DME",
        # Suspected
        "Possible": "M - Suspected DME",
        # Ungradable
        "Qualité de la photographie insuffisante": "Ungradable",
        "Non applicable": "Ungradable",
    }

    return mapping.get(value, "Ungradable")


def get_macular_severity(standardized_value):
    """Return numeric severity for sorting (0-4, 99 for ungradable)."""
    severity_map = {
        "M0 - No DME": 0,
        "M - Suspected DME": 1,
        "M1 - DME >1dd from fovea": 2,
        "M2 - DME ≤1dd from fovea": 3,
        "M2 - DME ≤1dd (OCT-/E-)": 3,
        "M6 - Treated DME": 4,
        "Ungradable": 99,
    }
    return severity_map.get(standardized_value, 99)


# =============================================================================
# 3. CREATE CLINICAL DIAGNOSIS
# =============================================================================


def get_diagnosis_code(retino, macula):
    if retino == "Ungradable" or macula == "Ungradable":
        return "99 - Ungradable"

    # Handle treated cases separately
    if "Treated" in retino or "Treated" in macula:
        if "PDR" in retino:
            return "14 - Post-treatment PDR"
        elif "Treated" in macula:
            return "15 - Post-treatment DME"
        else:
            return "14 - Post-treatment NPDR"

    # Determine macular threat status
    has_macular_threat = "M1" in macula or "M2" in macula
    has_suspected_macular = "Suspected" in macula

    # Extract retinopathy level
    if "R0" in retino:
        r_level = 0
    elif "R1" in retino:
        r_level = 1
    elif "R2" in retino:
        r_level = 2
    elif "R3" in retino:
        r_level = 3
    elif "R4" in retino:
        r_level = 4
    else:
        r_level = -1

    # Build diagnosis based on retinopathy level and macular status
    if r_level == 0:
        if not has_macular_threat and not has_suspected_macular:
            return "01 - No DR"
        elif has_suspected_macular:
            return "02 - No DR, suspected DME"
        else:
            return "03 - No DR, DME present"

    elif r_level == 1:
        if not has_macular_threat and not has_suspected_macular:
            return "04 - Mild NPDR without DME"
        elif has_suspected_macular:
            return "05 - Mild NPDR, suspected DME"
        else:
            return "06 - Mild NPDR with DME"

    elif r_level == 2:
        if not has_macular_threat and not has_suspected_macular:
            return "07 - Moderate NPDR without DME"
        elif has_suspected_macular:
            return "08 - Moderate NPDR, suspected DME"
        else:
            return "09 - Moderate NPDR with DME"

    elif r_level == 3:
        if not has_macular_threat and not has_suspected_macular:
            return "10 - Severe NPDR without DME"
        elif has_suspected_macular:
            return "11 - Severe NPDR, suspected DME"
        else:
            return "12 - Severe NPDR with DME"

    elif r_level == 4:
        if has_macular_threat or has_suspected_macular:
            return "13 - Suspected PDR with DME"
        else:
            return "13 - Suspected PDR without DME"

    return "99 - Unclassified"


def create_diagnosis(row):
    """Create a clinically meaningful diagnosis combining retinopathy and macular status."""
    results = {"Diagnosis_OD": None, "Diagnosis_OS": None}
    for laterality in ["OD", "OS"]:
        retino = row[f"Retinopathy_std_{laterality}"]
        macula = row[f"Macular_std_{laterality}"]

        diagnosis_code = get_diagnosis_code(retino, macula)
        results[f"Diagnosis_{laterality}"] = diagnosis_code

    return pd.Series(results)


def diagnosis_to_etdrs5(diagnosis_code: str) -> int:
    """Map a fine-grained diagnosis string to the ETDRS 5-stage scale.

    ETDRS 5 stages:
        0 - No DR
        1 - Mild NPDR
        2 - Moderate NPDR
        3 - Severe NPDR
        4 - PDR

    DME status is discarded (orthogonal axis in ETDRS).
    """
    prefix = diagnosis_code[:2]
    mapping = {
        "01": 0,
        "02": 0,
        "03": 0,  # No DR (± DME)
        "04": 1,
        "05": 1,
        "06": 1,  # Mild NPDR
        "07": 2,
        "08": 2,
        "09": 2,  # Moderate NPDR
        "10": 3,
        "11": 3,
        "12": 3,  # Severe NPDR
        "13": 4,  # PDR
    }
    return mapping.get(prefix, -1)


def diagnosis_to_screening_referability(diagnosis_code: str) -> int:
    """Map diagnosis to binary screening referability based on DR severity only.

    Since DME cannot be inferred from lesion segmentation alone,
    referability is based purely on retinopathy level:
        Non-referable (0): No DR or Mild NPDR (r_level 0-1)
        Referable (1): Moderate NPDR or worse (r_level 2+)
    """
    prefix = diagnosis_code[:2]
    non_referable = {
        "01",
        "02",
        "03",  # No DR (± DME)
        "04",
        "05",
        "06",  # Mild NPDR (± DME)
    }
    referable = {
        "07",
        "08",
        "09",  # Moderate NPDR (± DME)
        "10",
        "11",
        "12",  # Severe NPDR (± DME)
        "13",  # PDR
    }
    if prefix in non_referable:
        return 0
    elif prefix in referable:
        return 1
    return -1
