from abc import ABC, abstractmethod
from typing import Tuple, NewType


Event = NewType('Event', str)
EventToAlign = NewType('EventToAlign', str)


class ProcessingStrategy(ABC):

    @abstractmethod
    def process(self, data_preprocessor):
        pass


class BehaviorProcessingStrategy(ProcessingStrategy):
    def __init__(self, config_key: str, time_window: Tuple[int, int], events: Tuple[Tuple[Event, EventToAlign]], return_df: bool = False):
        self.config_key = config_key
        self.time_window = time_window
        self.events = events
        self.return_df = return_df

    def process(self, data_preprocessor):

        behavior_df = data_preprocessor.generate_df_from_config_dict(
            config_key='behavioral_events')

        mean_dict = data_preprocessor.batch_calculate_mean_event_frequency(
            behavior_df, self.time_window, *self.events)

        data_preprocessor.aggreate_processed_results(
            mean_dict, return_df=self.return_df)
