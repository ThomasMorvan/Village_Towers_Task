from village.custom_classes.custom_area_base import CustomAreaBase


class TowerCustomArea(CustomAreaBase):
    """T-shaped BOX area: corridor bar + vertical stem. Tune to your geometry
    (pixels in the CAM_BOX_RESOLUTION frame)."""

    name = "T_AREA"
    active = True
    threshold = 65

    polygons = [[[55, 215], [585, 215], [585, 275], [55, 275]],  # h
                [[300, 60], [360, 60], [360, 275], [300, 275]]]  # v
