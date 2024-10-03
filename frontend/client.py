import os
import json
import logging
import requests
from dds_utils import Results, Region
import yaml

class Client:
    def __init__(self, config, client_id, server_handle=None):
        self.client_id = client_id
        if server_handle:
            self.server = server_handle
        else:
            self.hname = server_handle
            self.session = requests.Session()

        self.config = config

        self.logger = logging.getLogger(f"client-{self.client_id}")
        handler = logging.NullHandler()
        self.logger.addHandler(handler)
        # Set the log format to include client_id
        formatter = logging.Formatter(f"%(name)s -- %(levelname)s -- {self.client_id} -- %(lineno)d -- %(message)s")
        handler.setFormatter(formatter)

        self.logger.info(f"Client {self.client_id} initialized")

    def init_server(self, nframes):
        self.config['nframes'] = nframes
        response = self.session.post(
            f"http://{self.hname}/init", data=yaml.dump(self.config))
        if response.status_code != 200:
            self.logger.fatal("Could not initialize server")
            exit()

    def get_first_phase_results(self, vid_name):
        encoded_vid_path = os.path.join(
            vid_name + "-base-phase-cropped", "temp.mp4")
        video_to_send = {"media": open(encoded_vid_path, "rb")}
        response = self.session.post(
            f"http://{self.hname}/low", files=video_to_send)
        response_json = json.loads(response.text)

        results = Results()
        for region in response_json["results"]:
            results.append(Region.convert_from_server_response(
                region, self.config.low_resolution, "low-res"))
        rpn = Results()
        for region in response_json["req_regions"]:
            rpn.append(Region.convert_from_server_response(
                region, self.config.low_resolution, "low-res"))

        return results, rpn

    def get_second_phase_results(self, vid_name):
        encoded_vid_path = os.path.join(vid_name + "-cropped", "temp.mp4")
        video_to_send = {"media": open(encoded_vid_path, "rb")}
        response = self.session.post(
            f"http://{self.hname}/high", files=video_to_send)
        response_json = json.loads(response.text)

        results = Results()
        for region in response_json["results"]:
            results.append(Region.convert_from_server_response(
                region, self.config.high_resolution, "high-res"))

        return results
