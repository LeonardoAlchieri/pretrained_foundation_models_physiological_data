from typing import TypeAlias

import numpy as np

DataInfoValue: TypeAlias = str | np.ndarray | list[str] | None
DataInfo: TypeAlias = dict[str, DataInfoValue]
