from frontend.roi_cache import RoICache
from frontend.client import Client
class RoIClient(Client):
    def __init__(self, config, client_id, server_handle=None):
        super().__init__(config, client_id, server_handle)
        self.roi_cache = RoICache(
            config['RoI_time_window'],
            config['RoI_cache_conf_threshold'],
            config['relevant_classes'],
            config['RoI_cache_residual_threshold'],
            config['low_resolution']
        )

    def add_results_to_cache(self, video_name, frame_id, results, frame_image):
        self.roi_cache.add_results(video_name, frame_id, results, frame_image)

    def get_regions_of_interest(self, current_frame_id, current_frame_image):
        return self.roi_cache.get_regions_of_interest(current_frame_id, current_frame_image)