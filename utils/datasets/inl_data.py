from collections import Counter
from dataclasses import dataclass, field
from typing import List
from itertools import chain
# #####################################
from .ItemInfo import ValidItem, InvalidItem
# #####################################


@dataclass
class InL_Data:
    stats:          Counter
    label_stats:    Counter
    valid_items:    List[ValidItem] = field(repr=False)
    invalid_items:  List[InvalidItem] = field(repr=False)
    # #####################################

    @property
    def n_files(self) -> int:
        return len(self.valid_items) + len(self.invalid_items)
    # #####################################

    @property
    def n_images(self) -> int:
        return len(self)
    # #####################################

    @property
    def n_labels(self) -> int:
        return sum(self.label_stats.values())
    # #####################################

    @property
    def labels(self):
        return list(chain([i.boxes for i in self.valid_items]))
    # #####################################

    @property
    def shapes(self):
        return list([i.shape for i in self.valid_items])
    # #####################################

    def __len__(self) -> int:           
        return len(self.valid_items)
    # #####################################

    def __getittem__(self, idx: int):
        return self.valid_items[idx]
    # #####################################

    def __iter__(self):
        return iter(self.valid_items)
    # #####################################
