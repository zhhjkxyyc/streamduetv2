from abc import ABC, abstractmethod
import logging
import os
import json
import shutil
from backend.server import Server
from frontend.client_factory import ClientFactory
from dds_utils import Results, Region, compute_regions_size, merge_boxes_in_results, cleanup, extract_images_from_video, read_results_dict
from streamduet_utils import list_frames,get_images_length,get_image_extension
from workspace.base_instance_strategy import InstanceStrategy
from workspace.streamduet_strategy import StreamDuetStrategy
from workspace.streamduet_RoI_strategy import StreamDuetRoIStrategy
class GTStrategy(InstanceStrategy):
    def run(self, args):
        self.logger.warning(f"Running GT in mode on {args.video_name}")
        if args.mode == "emulation":
            self.server = Server(self.config)
            self.client = ClientFactory.get_client(self.config, args.client_id, server=self.server)
        else:
            self.client = ClientFactory.get_client(self.config, args.client_id, hname=args.hname)

        results, bw = self.analyze_video_mpeg(args.video_name, args.high_images_path, args.enforce_iframes)
        return results, bw

class MPEGStrategy(InstanceStrategy):
    def run(self, args):
        self.logger.warning(f"Running in MPEG mode with resolution {args.low_resolution} on {args.video_name}")
        if args.mode == "emulation":
            self.server = Server(self.config)
            self.client = ClientFactory.get_client(self.config, args.client_id, server=self.server)
        else:
            self.client = ClientFactory.get_client(self.config, args.client_id, hname=args.hname)

        results, bw = self.analyze_video_mpeg(args.video_name, args.high_images_path, args.enforce_iframes)
        return results, bw
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

class DDSStrategy(InstanceStrategy):
    def run(self, args):
        self.logger.warning(f"Running StreamDuet in mode on {args.video_name}")
        if args.mode == "emulation":
            self.server = Server(self.config)
            self.client = ClientFactory.get_client(self.config, args.client_id, server=self.server)
            results, bw = self.analyze_video_emulate(
            args.video_name, args.high_images_path,
            args.enforce_iframes, args.low_results_path, args.debug_mode)
        else:
            self.client = ClientFactory.get_client(self.config, args.client_id, hname=args.hname)
            results, bw = self.analyze_video(args.video_name, args.high_images_path, args.enforce_iframes)

        return results, bw

    def analyze_video_emulate(self, video_name, high_images_path,
                              enforce_iframes, low_results_path=None,
                              debug_mode=False):
        final_results = Results()
        low_phase_results = Results()
        high_phase_results = Results()
        number_of_frames=get_images_length(high_images_path)
        # number_of_frames = len(
        #     [x for x in os.listdir(high_images_path) if "png" in x])

        low_results_dict = None
        if low_results_path:
            low_results_dict = read_results_dict(low_results_path)

        total_size = [0, 0]
        total_regions_count = 0
        for i in range(0, number_of_frames, self.config.batch_size):
            start_fid = i
            end_fid = min(number_of_frames, i + self.config.batch_size)
            self.logger.info(f"Processing batch from {start_fid} to {end_fid}")

            # Encode frames in batch and get size
            # Make temporary frames to downsize complete frames
            base_req_regions = Results()
            for fid in range(start_fid, end_fid):
                base_req_regions.append(
                    Region(fid, 0, 0, 1, 1, 1.0, 2,
                           self.config.high_resolution))
            encoded_batch_video_size, batch_pixel_size = compute_regions_size(
                base_req_regions, f"{video_name}-base-phase", high_images_path,
                self.config.low_resolution, self.config.low_qp,
                enforce_iframes, True)
            self.logger.info(f"Sent {encoded_batch_video_size / 1024} "
                             f"in base phase")
            total_size[0] += encoded_batch_video_size

            # Low resolution phase
            low_images_path = f"{video_name}-base-phase-cropped"
            r1, req_regions = self.server.simulate_low_query(
                start_fid, end_fid, low_images_path, low_results_dict, False,
                self.config.rpn_enlarge_ratio)
            total_regions_count += len(req_regions)

            low_phase_results.combine_results(
                r1, self.config.intersection_threshold)
            final_results.combine_results(
                r1, self.config.intersection_threshold)

            # High resolution phase
            if len(req_regions) > 0:
                # Crop, compress and get size

                regions_size, _ = compute_regions_size(
                    req_regions, video_name, high_images_path,
                    self.config.high_resolution, self.config.high_qp,
                    enforce_iframes, True)
                self.logger.info(f"Sent {len(req_regions)} regions which have "
                                 f"{regions_size / 1024}KB in second phase "
                                 f"using QP {self.config.high_qp}")
                total_size[1] += regions_size

                # High resolution phase every three filter
                r2 = self.server.simulate_high_query(
                    video_name, low_images_path, req_regions)
                self.logger.info(f"Got {len(r2)} results in second phase "
                                 f"of batch")

                high_phase_results.combine_results(
                    r2, self.config.intersection_threshold)
                final_results.combine_results(
                    r2, self.config.intersection_threshold)

            # Cleanup for the next batch
            cleanup(video_name, debug_mode, start_fid, end_fid)

        self.logger.info(f"Got {len(low_phase_results)} unique results "
                         f"in base phase")
        self.logger.info(f"Got {len(high_phase_results)} positive "
                         f"identifications out of {total_regions_count} "
                         f"requests in second phase")

        # Fill gaps in results
        final_results.fill_gaps(number_of_frames)

        # Write results
        final_results.write(f"{video_name}")

        self.logger.info(f"Writing results for {video_name}")
        self.logger.info(f"{len(final_results)} objects detected "
                         f"and {total_size[1]} total size "
                         f"of regions sent in high resolution")

        rdict = read_results_dict(f"{video_name}")
        final_results = merge_boxes_in_results(rdict, 0.3, 0.3)

        final_results.fill_gaps(number_of_frames)
        final_results.write(f"{video_name}")
        return final_results, total_size

    def analyze_video(self, vid_name, raw_images, config, enforce_iframes):
        final_results = Results()
        all_required_regions = Results()
        low_phase_size = 0
        high_phase_size = 0
        # nframes = sum(map(lambda e: "png" in e, os.listdir(raw_images)))
        nframes =list_frames(raw_images)
        # len(list_frames(raw_images))
        self.client.init_server(nframes)

        for i in range(0, nframes, self.config.batch_size):
            start_frame = i
            end_frame = min(nframes, i + self.config.batch_size)
            self.logger.info(f"Processing frames {start_frame} to {end_frame}")

            # First iteration
            req_regions = Results()
            for fid in range(start_frame, end_frame):
                req_regions.append(Region(fid, 0, 0, 1, 1, 1.0, 2, self.config.low_resolution))
            batch_video_size, _ =compute_regions_size(
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
            cleanup(vid_name, False, start_frame, end_frame)

        self.logger.info(f"Merging results")
        final_results = merge_boxes_in_results(
            final_results.regions_dict, 0.3, 0.3)
        self.logger.info(f"Writing results for {vid_name}")
        final_results.fill_gaps(nframes)

        final_results.combine_results(
            all_required_regions, self.config.intersection_threshold)

        final_results.write(f"{vid_name}")

        return final_results, (low_phase_size, high_phase_size)

class StrategyFactory:
    @staticmethod
    def get_strategy(args, config, logger):
        if args.method == 'gt':
            return GTStrategy(config, logger)
        elif args.method == 'mpeg':
            return MPEGStrategy(config, logger)
        elif args.method == 'dds':
            return DDSStrategy(config, logger)
        elif args.method == 'streamduet':
            return StreamDuetStrategy(config, logger)
        elif args.method == 'streamduetRoI':
            return StreamDuetRoIStrategy(config, logger)
        else:
            raise ValueError(f"Unknown method {args.method}")