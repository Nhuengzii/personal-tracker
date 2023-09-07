from enum import Enum
from src.models import osnet_ain, osnet

class AvailableEmbedderModels(Enum):
    OSNET_X1_0 = "osnet_x1_0"
    OSNET_AIN_X1_0 = "osnet_ain_x1_0"
