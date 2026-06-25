from village.custom_classes.custom_area_base import CustomAreaBase


class TowerCustomArea(CustomAreaBase):
    """T-shaped BOX area in camera coords."""

    name = "T_AREA"
    active = True
    threshold = 65

    polygons = [[[60, 225], [585, 225], [585, 265], [60, 265]],  # body bar
                [[585, 215], [625, 215], [625, 285], [585, 285]],  # stem tip
                [[15, 60], [60, 60], [60, 430], [15, 430]]]  # arms
