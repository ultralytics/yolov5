import logging
import torch
from abc import ABC, abstractmethod
from typing import Tuple
from ClassChannelsCreator import ClassChannelsCreator
from ThresholdingMaxConfidenceClassChannelsCreator import ThresholdingMaxConfidenceClassChannelsCreator

def create_class_channels_creator(option: str) -> ClassChannelsCreator:
    logger = logging.getLogger(__name__)
    if option == "ThresholdingMaxConfidenceClassChannelsCreator":
        logger.debug("ThresholdingMaxConfidenceClassChannelsCreator being used!")
        return ThresholdingMaxConfidenceClassChannelsCreator()
    else:
        # TODO: Add support for NoValueRegionAffixer
        logger.error("ClassChannelsCreator %s not supported!", option)
        raise Exception("ClassChannelsCreator type not supported")
