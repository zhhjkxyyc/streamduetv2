from abc import ABC, abstractmethod
import logging
import os
import json
import shutil
from backend.server import Server
from frontend.client_factory import ClientFactory
from dds_utils import Results, Region, compute_regions_size, merge_boxes_in_results, cleanup, extract_images_from_video, read_results_dict
from streamduet_utils import list_frames,get_images_length,get_image_extension

class InstanceStrategy(ABC):
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.server = None
        self.client = None

    @abstractmethod
    def run(self, args):
        pass

    def get_first_phase_results(self, vid_name):
        return self.client.get_first_phase_results(vid_name)

    def get_second_phase_results(self, vid_name):
        return self.client.get_second_phase_results(vid_name)

    def analyze_video(self, vid_name, raw_images, enforce_iframes):
        final_results = Results()
        all_required_regions = Results()
        low_phase_size = 0
        high_phase_size = 0
        nframes = sum(map(lambda e: "jpg" in e, os.listdir(raw_images)))

        self.client.init_server(nframes)

        for i in range(0, nframes, self.config.batch_size):
            start_frame = i
            end_frame = min(nframes, i + self.config.batch_size)
            self.logger.info(f"Processing frames {start_frame} to {end_frame}")

            # First iteration
            req_regions = Results()
            for fid in range(start_frame, end_frame):
                req_regions.append(Region(fid, 0, 0, 1, 1, 1.0, 2, self.config.low_resolution))
            batch_video_size, _ = compute_regions_size(
                req_regions, f"{vid_name}-base-phase", raw_images,
                self.config.low_resolution, self.config.low_qp,
                enforce_iframes, True)
            low_phase_size += batch_video_size
            self.logger.info(f"{batch_video_size / 1024}KB sent in base phase."
                             f"Using QP {self.config.low_qp} and "
                             f"Resolution {self.config.low_resolution}.")
            results, rpn_regions = self.get_first_phase_results(vid_name)
            final_results.combine_results(
                results, self.config.intersection_threshold)
            all_required_regions.combine_results(
                rpn_regions, self.config.intersection_threshold)

            # Second Iteration
            if len(rpn_regions) > 0:
                batch_video_size, _ = compute_regions_size(
                    rpn_regions, vid_name, raw_images,
                    self.config.high_resolution, self.config.high_qp,
                    enforce_iframes, True)
                high_phase_size += batch_video_size
                self.logger.info(f"{batch_video_size / 1024}KB sent in second "
                                 f"phase. Using QP {self.config.high_qp} and "
                                 f"Resolution {self.config.high_resolution}.")
                results = self.get_second_phase_results(vid_name)
                final_results.combine_results(
                    results, self.config.intersection_threshold)

            # Cleanup for the next batch
            cleanup(vid_name, self.config.debug_mode, start_frame, end_frame)

        self.logger.info(f"Merging results")
        final_results = merge_boxes_in_results(
            final_results.regions_dict, 0.3, 0.3)
        self.logger.info(f"Writing results for {vid_name}")
        final_results.fill_gaps(nframes)

        final_results.combine_results(
            all_required_regions, self.config.intersection_threshold)

        final_results.write(f"{vid_name}")

        return final_results, (low_phase_size, high_phase_size)

    def analyze_video_mpeg(self, video_name, raw_images_path, enforce_iframes):
        number_of_frames = len([f for f in os.listdir(raw_images_path) if ".jpg" in f])

        final_results = Results()
        final_rpn_results = Results()
        total_size = 0
        for i in range(0, number_of_frames, self.config.batch_size):
            start_frame = i
            end_frame = min(number_of_frames, i + self.config.batch_size)

            batch_fnames = sorted([f"{str(idx).zfill(8)}.jpg" for idx in range(start_frame, end_frame)])

            req_regions = Results()
            for fid in range(start_frame, end_frame):
                req_regions.append(Region(fid, 0, 0, 1, 1, 1.0, 2, self.config.low_resolution))
            batch_video_size, _ = compute_regions_size(req_regions, f"{video_name}-base-phase", raw_images_path,
                                                       self.config.low_resolution, self.config.low_qp, enforce_iframes, True)
            self.logger.info(f"{batch_video_size / 1024}KB sent in base phase using {self.config.low_qp}QP")
            extract_images_from_video(f"{video_name}-base-phase-cropped", req_regions)
            results, rpn_results = self.server.perform_detection(f"{video_name}-base-phase-cropped",
                                                                 self.config.low_resolution, batch_fnames)

            self.logger.info(f"Detection {len(results)} regions for batch {start_frame} to {end_frame} with a total size of {batch_video_size / 1024}KB")
            final_results.combine_results(results, self.config.intersection_threshold)
            final_rpn_results.combine_results(rpn_results, self.config.intersection_threshold)

            # Remove encoded video manually
            shutil.rmtree(f"{video_name}-base-phase-cropped")
            total_size += batch_video_size

        final_results = merge_boxes_in_results(final_results.regions_dict, 0.3, 0.3)
        final_results.fill_gaps(number_of_frames)

        # Add RPN regions
        final_results.combine_results(final_rpn_results, self.config.intersection_threshold)

        final_results.write(video_name)

        return final_results, [total_size, 0]
