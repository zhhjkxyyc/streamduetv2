import os
import shutil
import logging
from dds_utils import (Results, Region, calc_iou, merge_images,
                       extract_images_from_video, merge_boxes_in_results,
                       compute_area_of_frame, calc_area)
from backend.object_detector import Detector
from backend.image_processing import (perform_server_cleanup, reset_server_state, perform_detection, extract_images)
from streamduet_utils import list_frames,get_images_length,get_image_extension
import cv2 as cv
class BaseServer:
    def __init__(self, config, nframes=None):
        self.relevant_classes = ['vehicle', 'person', 'roadside-objects']
        self.config = config
        self.logger = logging.getLogger("server")
        handler = logging.NullHandler()
        self.logger.addHandler(handler)

        model_type = config['model_type']
        model_config = config['model']
        self.detector = Detector(model_config)

        self.curr_fid = 0
        self.nframes = nframes
        self.last_requested_regions = None
        self.client_id = config['client_id']
        self.logger.info(f"Server for client {self.client_id} started")

    def reset_state(self, nframes):
        reset_server_state(nframes)

    def perform_server_cleanup(self):
        perform_server_cleanup()

    def perform_detection(self, images_direc, resolution, fnames=None,
                          images=None):
        final_results = Results()
        rpn_regions = Results()

        if fnames is None:
            fnames = sorted(os.listdir(images_direc))
        self.logger.info(f"Running inference on {len(fnames)} frames")
        for fname in fnames:
            if "jpg" not in fname:
                continue
            fid = int(fname.split(".")[0])
            image = None
            if images:
                image = images[fid]
            else:
                image_path = os.path.join(images_direc, fname)
                image = cv.imread(image_path)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

            detection_results, rpn_results = (
                self.detector.infer(image))
            frame_with_no_results = True
            for label, conf, (x, y, w, h) in detection_results:
                if (self.config.min_object_size and
                        w * h < self.config.min_object_size) or w * h == 0.0:
                    continue
                r = Region(fid, x, y, w, h, conf, label,
                           resolution, origin="mpeg")
                final_results.append(r)
                frame_with_no_results = False
            for label, conf, (x, y, w, h) in rpn_results:
                r = Region(fid, x, y, w, h, conf, label,
                           resolution, origin="generic")
                rpn_regions.append(r)
                frame_with_no_results = False
            self.logger.debug(
                f"Got {len(final_results)} results "
                f"and {len(rpn_regions)} for {fname}")

            if frame_with_no_results:
                final_results.append(
                    Region(fid, 0, 0, 0, 0, 0.1, "no obj", resolution))

        return final_results, rpn_regions

    def get_regions_to_query(self, rpn_regions, detections):
        req_regions = Results()
        for region in rpn_regions.regions:
            # if region.w * region.h > self.config.size_obj:
            #     continue
            if len(detections) > 0:
                matches = 0
                for detection in detections.regions:
                    if (calc_iou(detection, region) > self.config.objfilter_iou and detection.fid == region.fid and region.label == 'object'):
                        matches += 1
                if matches > 0:
                    continue
            region.enlarge(self.config.rpn_enlarge_ratio)
            req_regions.add_single_result(region, self.config.intersection_threshold)
        return req_regions

    def simulate_low_query(self, start_fid, end_fid, images_direc,
                           results_dict, simulation=True,
                           rpn_enlarge_ratio=0.0, extract_regions=True):
        if extract_regions:
            # If called from actual implementation
            # This will not run
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
        # Divide RPN results into detections and RPN regions
        for single_result in batch_results.regions:
            if (single_result.conf > self.config.prune_score and
                    single_result.label in self.relevant_classes):
                detections.add_single_result(
                    single_result, self.config.intersection_threshold)
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
        image_extension=get_image_extension(low_images_direc)
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
        for r in results_with_detections_only.regions:
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

        shutil.rmtree(merged_images_direc)
        return results_with_detections_only

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

    def perform_high_query(self, file_data):
        low_images_direc = f"server_temp_{self.client_id}"
        cropped_images_direc = f"server_temp_{self.client_id}-cropped"

        with open(os.path.join(cropped_images_direc, "temp.mp4"), "wb") as f:
            f.write(file_data.read())

        results = self.simulate_high_query(low_images_direc, low_images_direc, self.last_requested_regions)

        results_list = []
        for r in results.regions:
            results_list.append([r.fid, r.x, r.y, r.w, r.h, r.conf, r.label])

        self.perform_server_cleanup()
        return {
            "results": results_list,
            "req_region": []
        }
