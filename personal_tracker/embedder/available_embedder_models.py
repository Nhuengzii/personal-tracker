from enum import Enum
from personal_tracker.models import osnet_ain, osnet

class AvailableEmbedderModels(Enum):
    OSNET_X1_0 = "osnet_x1_0"
    OSNET_X0_75 = "osnet_x0_75"
    OSNET_X0_5 = "osnet_x0_5"
    OSNET_X0_25 = "osnet_x0_25"
    OSNET_AIN_X1_0 = "osnet_ain_x1_0"
    OSNET_AIN_X0_75 = "osnet_ain_x0_75"
    OSNET_AIN_X0_5 = "osnet_ain_x0_5"
    OSNET_AIN_X0_25 = "osnet_ain_x0_25"
    CLIP = "clip"
