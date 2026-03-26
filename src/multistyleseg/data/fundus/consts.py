from enum import Flag, auto


class Lesions(str, Flag):
    COTTON_WOOL_SPOT = auto()
    EXUDATES = auto()
    HEMORRHAGES = auto()
    MICROANEURYSMS = auto()

    @property
    def name(cls):
        name = super(Lesions, cls).name
        if name:
            return name
        else:
            return [flag.name for flag in Lesions if flag in cls]

    @property
    def str_name(cls):
        name = cls.name
        if isinstance(name, list):
            return "_".join(name)
        else:
            return name

    @property
    def length(cls):
        name = cls.name
        if isinstance(name, list):
            return len(name)
        else:
            return 1

    def __len__(self):
        return self.length


ALL_CLASSES = [
    Lesions.COTTON_WOOL_SPOT,
    Lesions.EXUDATES,
    Lesions.HEMORRHAGES,
    Lesions.MICROANEURYSMS,
]


MAPPING = {
    Lesions.COTTON_WOOL_SPOT: "Cotton\nWool\nSpot",
    Lesions.EXUDATES: "Exudates",
    Lesions.HEMORRHAGES: "Hemorrhages",
    Lesions.MICROANEURYSMS: "Microaneurysms",
}

MAPPING_STR = {
    "COTTON_WOOL_SPOT": "CWS",
    "EXUDATES": "EX",
    "HEMORRHAGES": "HEM",
    "MICROANEURYSMS": "μA",
}


TEST_DATASET_SIZE = {
    "IDRID": 27,
    "FGADR": 369,
    "MESSIDOR": 60,
    "DDR": 225,
    "RETLES": 319,
}
