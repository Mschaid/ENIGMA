import numpy as np
import pandas as pd
import tensorflow as tf


class SequenceProcessor:
    def __init__(self, data):
        self.data = data
