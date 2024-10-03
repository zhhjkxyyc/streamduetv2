import os
import shutil
import logging
import cv2
from dds_utils import (Results, Region, calc_iou, merge_images,
                       extract_images_from_video, merge_boxes_in_results,
                       compute_area_of_frame, calc_area)
from backend.object_detector import Detector
from backend.image_processing import (perform_server_cleanup, reset_server_state, perform_detection, extract_images)
from backend.base_server import BaseServer
from backend.inferenc_cache import InferenceCache
from streamduet_utils import list_frames,get_images_length,get_image_extension
from backend.server import BaseServer
import time
class RoICacheServer(BaseServer):
    def __init__(self, config, nframes=None):
        super().__init__(config, nframes)
        self.cache = InferenceCache(config['time_window'], config['cache_dir'], config['cache_conf_threshold'])

    def save_high_conf_results(self, video_name, frame_id, results, frame_image):
        high_conf_results = [result for result in results if result.conf > self.config.high_threshold]
        self.cache.add_results(video_name, frame_id, high_conf_results, frame_image)

    def match_with_cache(self, video_name, results, frame_image):
        for result in results:
            if result.conf < self.config.low_threshold:
                best_result, best_image_path = self.cache.get_best_result(result, frame_image)
                if best_result:
                    result.label = best_result.label
                    result.conf = best_result.conf

    def perform_low_query(self, vid_data):
        extract_images(vid_data)

        start_fid = self.curr_fid
        end_fid = min(self.curr_fid + self.config.batch_size, self.nframes)
        self.logger.info(f"Processing frames from {start_fid} to {end_fid}")
        req_regions = Results()
        for fid in range(start_fid, end_fid):
            req_regions.append(Region(fid, 0, 0, 1, 1, 1.0, 2, self.config.low_resolution))
        extract_images_from_video(f"server_temp_{self.client_id}", req_regions)
        fnames = [f for f in os.listdir(f"server_temp_{self.client_id}") if "jpg" in f]

        results, rpn = self.perform_detection(f"server_temp_{self.client_id}", self.config.low_resolution, fnames)

        # Read frame image
        frame_image_path = os.path.join(f"server_temp_{self.client_id}", fnames[0])
        frame_image = cv2.imread(frame_image_path)

        # Match with cache
        self.match_with_cache(vid_data['video_name'], results.regions, frame_image)

        # Save high confidence results with frame image
        self.save_high_conf_results(vid_data['video_name'], start_fid, results.regions, frame_image)

        batch_results = Results()
        batch_results.combine_results(results, self.config.intersection_threshold)
        batch_results = merge_boxes_in_results(batch_results.regions_dict, 0.3, 0.3)
        batch_results.combine_results(rpn, self.config.intersection_threshold)

        detections, regions_to_query = self.simulate_low_query(start_fid, end_fid, f"server_temp_{self.client_id}", batch_results.regions_dict, False, self.config.rpn_enlarge_ratio, False)

        self.last_requested_regions = regions_to_query
        self.curr_fid = end_fid

        detections_list = []
        for r in detections.regions:
            detections_list.append([r.fid, r.x, r.y, r.w, r.h, r.conf, r.label])
        req_regions_list = []
        for r in regions_to_query.regions:
            req_regions_list.append([r.fid, r.x, r.y, r.w, r.h, r.conf, r.label])

        return {
            "results": detections_list,
            "req_regions": req_regions_list
        }

    def simulate_low_query(self, start_fid, end_fid, images_direc, results_dict, video_name,simulation=True, rpn_enlarge_ratio=0.0, extract_regions=True,base_req_regions=None):

        if extract_regions:
            base_req_regions = Results()
            for fid in range(start_fid, end_fid):
                base_req_regions.append(
                    Region(fid, 0, 0, 1, 1, 1.0, 2,
                            self.config.high_resolution))
            extract_images_from_video(images_direc, base_req_regions)

        batch_results = Results()

        self.logger.info(f"Getting results with threshold "
                         f"{self.config.low_threshold} and "
                         f"{self.config.high_threshold}")
        # Extract relevant results
        for fid in range(start_fid, end_fid):
            fid_results = results_dict[fid]
            for single_result in fid_results:
                single_result.origin = "low-res"
                batch_results.add_single_result(
                    single_result, self.config.intersection_threshold)

        detections = Results()
        rpn_regions = Results()
        frame_images = {fid: cv2.imread(self._get_image_path(images_direc, fid)) for fid in range(start_fid, end_fid)}

        # Divide RPN results into detections and RPN regions
        for single_result in batch_results.regions:
            if single_result.conf > self.config.high_threshold:
                if single_result.label in self.relevant_classes:
                    detections.add_single_result(
                        single_result, self.config.intersection_threshold)
                    frame_image = frame_images[single_result.fid]
                    self.cache.add_results(video_name, single_result.fid, [single_result], frame_image)
                else:
                    continue
            else:
                frame_image = frame_images[single_result.fid]
                best_result, _ = self.cache.get_best_result(single_result, frame_image)
                if best_result:
                    detections.add_single_result(
                        best_result, self.config.intersection_threshold)
                    self.cache.add_results(video_name, single_result.fid, [best_result], frame_image)
                else:
                    rpn_regions.add_single_result(
                        single_result, self.config.intersection_threshold)

        regions_to_query = self.get_regions_to_query(rpn_regions, detections)

        return detections, regions_to_query

    def simulate_high_query(self, vid_name, low_images_direc, req_regions):
        images_direc = f"{vid_name}-cropped"
        extract_images_from_video(images_direc, req_regions)

        if not os.path.isdir(images_direc):
            self.logger.error("Images directory was not found but the second iteration was called anyway")
            return Results()

        image_extension = get_image_extension(low_images_direc)
        fnames = sorted([f for f in os.listdir(images_direc) if image_extension in f])
        merged_images_direc = os.path.join(images_direc, "merged")
        os.makedirs(merged_images_direc, exist_ok=True)
        for img in fnames:
            shutil.copy(os.path.join(images_direc, img), merged_images_direc)

        merged_images = merge_images(merged_images_direc, low_images_direc, req_regions)
        results, _ = self.perform_detection(merged_images_direc, self.config.high_resolution, fnames, merged_images)

        results_with_detections_only = Results()
        for r in results.regions:
            if r.label == "no obj":
                continue
            results_with_detections_only.add_single_result(r, self.config.intersection_threshold)

        high_only_results = Results()
        area_dict = {}
        frame_images = {r.fid: cv2.imread(self._get_image_path(low_images_direc, r.fid)) for r in req_regions.regions}

        for r in results_with_detections_only.regions:
            if not r.fid in req_regions.regions_dict:
                continue
            frame_regions = req_regions.regions_dict[r.fid]
            regions_area = 0
            if r.fid in area_dict:
                regions_area = area_dict[r.fid]
            else:
                regions_area = compute_area_of_frame(frame_regions)
                area_dict[r.fid] = regions_area
            regions_with_result = frame_regions + [r]
            total_area = compute_area_of_frame(regions_with_result)
            extra_area = total_area - regions_area
            if extra_area < 0.05 * calc_area(r):
                r.origin = "high-res"
                high_only_results.append(r)

            # Save high confidence results to cache
            frame_image = frame_images[r.fid]
            if r.conf > self.config.high_threshold:
                self.cache.add_results(vid_name, r.fid, [r], frame_image)
            else:
                # Match with cache
                best_result, _ = self.cache.get_best_result(r, frame_image)
                if best_result:
                    best_result.origin = "high-cache-res"
                    high_only_results.add_single_result(best_result, self.config.intersection_threshold)
                    self.cache.add_results(vid_name, r.fid, [best_result], frame_image)

        shutil.rmtree(merged_images_direc)
        return high_only_results
    def _get_image_path(self, images_direc, frame_id):
        return os.path.join(images_direc,f"{str(frame_id).zfill(8)}.jpg")