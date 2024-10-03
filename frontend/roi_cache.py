import time
import pandas as pd
from collections import deque
import os
import cv2
import numpy as np
from dds_utils import Results, Region
from streamduet_utils import draw_bboxes_on_image
from concurrent.futures import ThreadPoolExecutor
class RoICache:
    def __init__(self, time_window, conf_threshold, relevant_classes, residual_threshold, lowres_threshold):
        self.time_window = time_window
        self.conf_threshold = conf_threshold
        self.relevant_classes = relevant_classes
        self.residual_threshold = residual_threshold
        self.lowres_threshold = lowres_threshold
        self.sift = cv2.SIFT_create()
        self.memory_cache = {}  # 内存缓存字典，按 frame_id 分组

    def _current_time(self):
        return time.time()

    def add_results(self, frame_id, results, frame_image):
        # 检查内存缓存中是否已有该帧结果和图像
        if frame_id not in self.memory_cache:
            # 保存帧结果和图像到内存缓存
            categorized_results = self._categorize_results(results)
            self.memory_cache[frame_id] = {
                'results': categorized_results,
                'image': frame_image,
                'timestamp': self._current_time()
            }

        # 清理过期的帧结果和图像
        self._cleanup()

    def get_regions_of_interest(self, current_frame_id, current_frame_image):
        prev_frame = self._get_previous_frame(current_frame_id)


        if prev_frame is None:
            req_regions = Results()
            background_regions = Results()
            current_frame_results = Results()
            req_regions.append(Region(current_frame_id, 0, 0, 1, 1, 1.0, 2, self.lowres_threshold))
            return req_regions, background_regions, current_frame_results

        roi_regions, background_regions, current_frame_results = self._process_cached_results(
            current_frame_id, current_frame_image)

        # draw_bboxes_on_image(roi_regions, current_frame_image, f"results/roi_regions-{current_frame_id}.jpg")
        # draw_bboxes_on_image(background_regions, current_frame_image, f"results/background_regions-{current_frame_id}.jpg")
        # draw_bboxes_on_image(current_frame_results, current_frame_image, f"results/current_frame_results-{current_frame_id}.jpg")
        roi_regions.combine_results(current_frame_results)

        return roi_regions, background_regions, current_frame_results

    def _process_cached_results(self, current_frame_id, current_frame_image):
        roi_regions = Results()
        background_regions = Results()
        current_frame_results = Results()
        # 这里改成16或32试试，16会很慢，换一个方式
        # predict_motion_results=self.process_frames(current_frame_image, current_frame_id, 75, 16,32)
        # draw_bboxes_on_image(predict_motion_results, current_frame_image, f"results/predict_motion_results-{current_frame_id}.jpg")
        for frame_id, data in self.memory_cache.items():
            if frame_id < current_frame_id - self.time_window:
                continue
            categorized_results = data['results']
            for category, results in categorized_results.items():

                # self.memory_cache中存放了之前不同帧的id和对应图像数据。我想实现功能，根据之前不同id的帧计算光流，预测当前帧的图像，考虑之前帧id的差值，考虑当前帧与之前帧的id的差值。然后与当前帧对比，并计算残差，输出残差大于阈值的宏块对等于BBOX，为Result。其中宏块大小可以预设。下面我给出Result类的定义：




                predicted_bboxes = self.predict_bounding_boxes(data['image'], current_frame_image, results,current_frame_id)
                predicted_bboxes.suppress()
                # current_frame_results.combine_results(predicted_bboxes, self.conf_threshold)
                self._categorize_predicted_bboxes(predicted_bboxes, category,current_frame_results, roi_regions, background_regions,current_frame_id)
        current_frame_results.suppress()
        # draw_bboxes_on_image(roi_regions, current_frame_image, f"results/roi_regions-withoutresual-{current_frame_id}.jpg")
        # roi_regions.combine_results(predict_motion_results)
        # draw_bboxes_on_image(current_frame_results, current_frame_image,
        #                      f"results/predict_motion_results-{current_frame_id}.jpg")
        return roi_regions, background_regions, current_frame_results

    def _categorize_predicted_bboxes(self, predicted_bboxes, category, current_frame_results,roi_regions, background_regions,current_frame_id):
        for predicted_bbox in predicted_bboxes.regions:
            predicted_bbox.frame_id=current_frame_id
            if category == 'high_conf_target' and predicted_bbox.conf > self.conf_threshold:
                current_frame_results.add_single_result(predicted_bbox, self.conf_threshold)
            elif category == 'high_conf_non_target' and predicted_bbox.conf > self.conf_threshold:
                background_regions.add_single_result(predicted_bbox, self.conf_threshold)
            elif category == 'low_conf':
                if predicted_bbox.conf > self.conf_threshold:
                    roi_regions.add_single_result(predicted_bbox, self.conf_threshold)
                else:
                    roi_regions.add_single_result(self._expand_bbox(predicted_bbox), self.conf_threshold)

    # 该函数预测的位置不准确，可否再结合残差校准预测的框
    def predict_bounding_boxes(self, prev_frame, current_frame_image, results, current_frame_id, threshold=30,
                               expand_pixels=10):
        # 将图像转换为灰度图像
        prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        current_frame_gray = cv2.cvtColor(current_frame_image, cv2.COLOR_BGR2GRAY)

        # 确保两个图像的尺寸匹配
        if prev_frame_gray.shape != current_frame_gray.shape:
            raise ValueError("Previous frame and current frame have different sizes.")

        predicted_bboxes = Results()
        flow = cv2.calcOpticalFlowFarneback(prev_frame_gray, current_frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        height, width = prev_frame_gray.shape[:2]

        for bbox in results.regions:
            x_min = int(bbox.x * width)
            y_min = int(bbox.y * height)
            x_max = int((bbox.x + bbox.w) * width)
            y_max = int((bbox.y + bbox.h) * height)

            flow_x = np.mean(flow[y_min:y_max, x_min:x_max, 0])
            flow_y = np.mean(flow[y_min:y_max, x_min:x_max, 1])

            # 预测的新位置
            new_x = bbox.x + flow_x / width
            new_y = bbox.y + flow_y / height

            # 计算残差并校准
            residual_block = current_frame_gray[y_min:y_max, x_min:x_max] - prev_frame_gray[y_min:y_max, x_min:x_max]
            mean_residual = np.mean(np.abs(residual_block))

            if mean_residual > threshold:
                # 如果残差大于阈值，进行校准
                residual_x = np.mean(np.sign(residual_block) * flow[y_min:y_max, x_min:x_max, 0])
                residual_y = np.mean(np.sign(residual_block) * flow[y_min:y_max, x_min:x_max, 1])

                new_x += residual_x / width
                new_y += residual_y / height

            # 扩大框以增加冗余性
            x_min_new = max(int(new_x * width) - expand_pixels, 0)
            y_min_new = max(int(new_y * height) - expand_pixels, 0)
            x_max_new = min(int((new_x + bbox.w) * width) + expand_pixels, width)
            y_max_new = min(int((new_y + bbox.h) * height) + expand_pixels, height)

            new_bbox = Region(
                current_frame_id,
                x=x_min_new / width,
                y=y_min_new / height,
                w=(x_max_new - x_min_new) / width,
                h=(y_max_new - y_min_new) / height,
                conf=bbox.conf,
                label=bbox.label,
                resolution=bbox.resolution,
                origin="backend_predicted"
            )
            predicted_bboxes.add_single_result(new_bbox)

        return predicted_bboxes

    def _expand_bbox(self, bbox):
        expanded_bbox = Region(
            bbox.fid,
            x=max(0, bbox.x - 0.01),
            y=max(0, bbox.y - 0.01),
            w=min(1, bbox.w*1.1),
            h=min(1, bbox.h*1.1),
            conf=bbox.conf,
            label=bbox.label,
            resolution=bbox.resolution,
            origin=bbox.origin
        )
        return expanded_bbox

    def _compute_moving_blocks(self, roi_regions, prev_frame, current_frame_image, current_frame_id):
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(current_frame_image, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        h, w = curr_gray.shape
        block_size = 64
        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                # Ensure the block includes the boundary
                y_end = min(y + block_size, h)
                x_end = min(x + block_size, w)
                flow_block = flow[y:y_end, x:x_end]
                mean_flow = np.mean(flow_block, axis=(0, 1))
                if np.linalg.norm(mean_flow) > self.residual_threshold:
                    roi_regions.add_single_result(
                        Region(current_frame_id, x / w, y / h, (x_end - x) / w, (y_end - y) / h, 0, 'moving_block', 1,
                               'motion'),
                        self.conf_threshold)

    def _categorize_results(self, results):
        categorized_results = {
            'high_conf_target': Results(),
            'high_conf_non_target':  Results(),
            'low_conf':   Results()
        }
        # 'Results' object is not iterable
        for region in results:
            if region.conf > self.conf_threshold:
                if region.label in self.relevant_classes:
                    categorized_results['high_conf_target'].append(region)
                else:
                    categorized_results['high_conf_non_target'].append(region)
            else:
                categorized_results['low_conf'].append(region)
        return categorized_results

    def _cleanup(self):
        min_frame_id = max(self.memory_cache.keys()) - self.time_window
        keys_to_delete = [frame_id for frame_id in self.memory_cache if frame_id < min_frame_id]
        for frame_id in keys_to_delete:
            del self.memory_cache[frame_id]

    def _get_previous_frame(self, current_frame_id):
        # 获取所有小于 current_frame_id 的 frame_id
        previous_frame_ids = [frame_id for frame_id in self.memory_cache if frame_id < current_frame_id]

        if not previous_frame_ids:
            return None

        # 获取小于 current_frame_id 的最大 frame_id
        prev_frame_id = max(previous_frame_ids)

        return self.memory_cache[prev_frame_id]['image']

    def _crop_image(self, image, bbox):
        height, width = image.shape[:2]
        x_min = int(bbox.x * width)
        y_min = int(bbox.y * height)
        x_max = int((bbox.x + bbox.w) * width)
        y_max = int((bbox.y + bbox.h) * height)
        return image[y_min:y_max, x_min:x_max]

    def _match_features(self, query_image, cached_image):
        query_keypoints, query_descriptors = self.sift.detectAndCompute(query_image, None)
        cached_keypoints, cached_descriptors = self.sift.detectAndCompute(cached_image, None)

        if query_descriptors is None or cached_descriptors is None:
            return False

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(query_descriptors, cached_descriptors, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        return len(good_matches) > 10  # Threshold for considering a match

    def draw_bbox_on_mask(self, mask, bbox):
        height, width = mask.shape[:2]
        x_min = int(bbox.x * width)
        y_min = int(bbox.y * height)
        x_max = int((bbox.x + bbox.w) * width)
        y_max = int((bbox.y + bbox.h) * height)
        mask[y_min:y_max, x_min:x_max] = 255
        return mask

    def is_region_in_mask(self, region, mask):
        height, width = mask.shape[:2]
        x_min = int(region.x * width)
        y_min = int(region.y * height)
        x_max = int((region.x + region.w) * width)
        y_max = int((region.y + region.h) * height)
        return np.any(mask[y_min:y_max, x_min:x_max] == 255)

    def process_frame(self, current_frame_id, current_frame_image):
        roi_regions, background_regions, current_frame_results = self.get_regions_of_interest(
            current_frame_id, current_frame_image)

        final_results = Results()
        final_results.combine_results(current_frame_results, self.conf_threshold)
        # combined_regions = background_regions.regions + current_frame_results.regions

        # draw_bboxes_on_image(roi_regions, current_frame_image, f"results/final_roi_regions-{current_frame_id}.jpg")

        # combined_mask = np.zeros(current_frame_image.shape[:2], dtype=np.uint8)
        # for region in combined_regions:
        #     combined_mask = self.draw_bbox_on_mask(combined_mask, region)
        #
        # base_req_regions = Results()
        # for region in roi_regions.regions:
        #     if not self.is_region_in_mask(region, combined_mask):
        #         base_req_regions.append(region)

        return roi_regions, final_results

    def update_cache(self, start_fid, end_fid, final_results, high_images_path):
        for fid in range(start_fid, end_fid):
            if fid in final_results.regions_dict:
                results= final_results.regions_dict[fid]
            else:
                results=[]
            if fid not in self.memory_cache:
                current_frame_image = cv2.imread(os.path.join(high_images_path, f"{fid:08d}.jpg"))
                self.add_results(fid, results, current_frame_image)
            else:
                self.add_results(fid, results, self.memory_cache[fid]['image'])
        self._cleanup()

    # def compute_optical_flow_farneback(self, prev_img, next_img):
    #     prev_img = self.ensure_grayscale(prev_img)
    #     next_img = self.ensure_grayscale(next_img)
    #     if prev_img.shape != next_img.shape:
    #         next_img = self.resize_image(next_img, (prev_img.shape[1], prev_img.shape[0]))
    #     flow = cv2.calcOpticalFlowFarneback(prev_img, next_img, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    #     return flow
    def compute_optical_flow_farneback(self,image1, image2):
        gray1 = self.ensure_grayscale(image1)
        gray2 = self.ensure_grayscale(image2)
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        motion_vectors = np.dstack((flow[..., 0], flow[..., 1]))
        return motion_vectors

    def resize_image(self,image, target_size):
        return cv2.resize(self,image, target_size)

    def ensure_grayscale(self,image):
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def warp_image(self,img, flow, scale_factor):
        h, w = img.shape
        flow_map = np.zeros_like(flow)
        flow_map[..., 0] = np.meshgrid(np.arange(w), np.arange(h))[0]
        flow_map[..., 1] = np.meshgrid(np.arange(w), np.arange(h))[1]
        flow_map += flow * scale_factor
        flow_map = flow_map.astype(np.float32)
        warped_img = cv2.remap(img, flow_map, None, cv2.INTER_LINEAR)
        return warped_img

        flow_map = np.zeros_like(flow)
        flow_map[..., 0] = np.meshgrid(np.arange(w), np.arange(h))[0]
        flow_map[..., 1] = np.meshgrid(np.arange(w), np.arange(h))[1]
        flow_map += flow
        flow_map = flow_map.astype(np.float32)
        warped_img = cv2.remap(img, flow_map, None, cv2.INTER_LINEAR)
        return warped_img

    def highlight_large_errors(self, img, residual, threshold, block_size, region_size, current_frame_id, expand_pixels=5):
        if len(residual.shape) == 3:
            h, w, _ = residual.shape
        else:
            h, w = residual.shape

        mask = np.zeros_like(img)
        highlighted_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        regions = []

        # 遍历图像，每次移动 region_size 大小，以避免重叠
        for y in range(0, h, region_size):
            for x in range(0, w, region_size):
                region_has_large_error = False

                # 遍历 region 内部的所有 block
                for by in range(y, min(y + region_size, h), block_size):
                    for bx in range(x, min(x + region_size, w), block_size):
                        block_h = min(block_size, h - by)
                        block_w = min(block_size, w - bx)

                        block = residual[by:by + block_h, bx:bx + block_w]
                        mean_error = np.mean(block)

                        # 如果块的平均误差超过阈值，标记该大区域并跳出内部循环
                        if mean_error > threshold:
                            region_has_large_error = True
                            mask[by:by + block_h, bx:bx + block_w] = 255
                            overlay = highlighted_img[by:by + block_h, bx:bx + block_w]
                            highlighted_img[by:by + block_h, bx:bx + block_w] = cv2.addWeighted(overlay, 0.5,
                                                                                                np.full_like(overlay,
                                                                                                             [0, 0,
                                                                                                              255]),
                                                                                                0.5, 0)
                            # 只要有一个 block 超过阈值，整个大区域将被标记，因此跳出 block 的遍历
                            break
                    if region_has_large_error:
                        break

                # 如果该大区域内有任何小块误差超过阈值，加入该大区域
                if region_has_large_error:
                    # 计算扩展后的区域宽度和高度
                    expanded_x = max(0, x - expand_pixels)
                    expanded_y = max(0, y - expand_pixels)
                    expanded_w = min(region_size + 2 * expand_pixels, w - expanded_x)
                    expanded_h = min(region_size + 2 * expand_pixels, h - expanded_y)

                    # 将大区域加入到 regions 列表
                    regions.append(Region(
                        current_frame_id,
                        expanded_x / w,  # 扩展后区域的左上角 x 坐标归一化
                        expanded_y / h,  # 扩展后区域的左上角 y 坐标归一化
                        expanded_w / w,  # 扩展后区域宽度归一化
                        expanded_h / h,  # 扩展后区域高度归一化
                        1,  # 优先级
                        mean_error,  # 误差值
                        1  # 固定标识符
                    ))

        return highlighted_img, mask, regions

    # def highlight_large_errors(self, img, residual, threshold, block_size, current_frame_id):
    #     if len(residual.shape) == 3:
    #         h, w, _ = residual.shape
    #     else:
    #         h, w = residual.shape
    #
    #     mask = np.zeros_like(img)
    #     highlighted_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    #     regions = []
    #
    #     # 遍历图像，每次移动 block_size 大小，以避免重叠
    #     for y in range(0, h, block_size):
    #         for x in range(0, w, block_size):
    #             # 确保块不超过图像边界
    #             block_h = min(block_size, h - y)
    #             block_w = min(block_size, w - x)
    #
    #             block = residual[y:y + block_h, x:x + block_w]
    #             mean_error = np.mean(block)
    #
    #             # 如果块的平均误差超过阈值，记录该块为需要标记的区域
    #             if mean_error > threshold:
    #                 mask[y:y + block_h, x:x + block_w] = 255
    #                 overlay = highlighted_img[y:y + block_h, x:x + block_w]
    #                 highlighted_img[y:y + block_h, x:x + block_w] = cv2.addWeighted(overlay, 0.5,
    #                                                                                 np.full_like(overlay, [0, 0, 255]),
    #                                                                                 0.5, 0)
    #                 # 添加区域到 regions 列表，确保区域与块大小一致，无重叠
    #                 regions.append(Region(
    #                     current_frame_id,
    #                     x / w,  # 坐标归一化
    #                     y / h,  # 坐标归一化
    #                     block_w / w,  # 块宽度归一化
    #                     block_h / h,  # 块高度归一化
    #                     1,  # 优先级
    #                     mean_error,  # 误差值
    #                     1  # 固定标识符
    #                 ))
    #
    #     return highlighted_img, mask, regions

    # 老版本20241001
    # def highlight_large_errors(self, img, residual, threshold, block_size, current_frame_id,expand_pixels=48):
    #     if len(residual.shape) == 3:
    #         h, w, _ = residual.shape
    #     else:
    #         h, w = residual.shape
    #
    #     mask = np.zeros_like(img)
    #     highlighted_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    #     regions = []
    #
    #     for y in range(0, h, block_size):
    #         for x in range(0, w, block_size):
    #             block = residual[y:y + block_size, x:x + block_size]
    #             mean_error = np.mean(block)
    #             if mean_error > threshold:
    #                 mask[y:y + block_size, x:x + block_size] = 255
    #                 overlay = highlighted_img[y:y + block_size, x:x + block_size]
    #                 highlighted_img[y:y + block_size, x:x + block_size] = cv2.addWeighted(overlay, 0.5,
    #                                                                                       np.full_like(overlay,
    #                                                                                                    [0, 0, 255]),
    #                                                                                       0.5, 0)
    #
    #                 # 扩大框以增加冗余性
    #                 x_min = max(x - expand_pixels, 0)
    #                 y_min = max(y - expand_pixels, 0)
    #                 x_max = min(x + block_size + expand_pixels, w)
    #                 y_max = min(y + block_size + expand_pixels, h)
    #                 # 归一化 x, y, block_size
    #                 regions.append(Region(
    #                     current_frame_id,
    #                     x_min / w,
    #                     y_min / h,
    #                     (x_max - x_min) / w,
    #                     (y_max - y_min) / h,
    #                     1,
    #                     mean_error,
    #                     1
    #                 ))
    #
    #     return highlighted_img, mask, regions



    def process_frames(self, current_img, current_frame_id, threshold=25, block_size=16,region_size=32):
        results = Results()
        time_stats = {}  # 用于存储每个步骤的时间统计

        # 统计开始时间
        start_time = time.time()

        # 第一步：获取帧ID并排序
        frame_ids = sorted(self.memory_cache.keys())
        if len(frame_ids) < 3:
            return results
        step_time = time.time() - start_time
        time_stats['frame_sorting'] = step_time

        # 第二步：取最大的和第二大的帧ID
        prev_frame_id = frame_ids[-1]
        prev_prev_frame_id = frame_ids[-2]

        if prev_frame_id >= current_frame_id or prev_prev_frame_id >= current_frame_id:
            return results

        # 第三步：将图像转换为灰度图
        start_time = time.time()
        prev_prev_img = self.ensure_grayscale(self.memory_cache[prev_prev_frame_id]["image"])
        prev_img = self.ensure_grayscale(self.memory_cache[prev_frame_id]["image"])
        current_img = self.ensure_grayscale(current_img)
        step_time = time.time() - start_time
        time_stats['grayscale_conversion'] = step_time

        # 第四步：调整图像大小
        start_time = time.time()
        if prev_img.shape != current_img.shape:
            prev_img = cv2.resize(prev_img, (current_img.shape[1], current_img.shape[0]))
        if prev_prev_img.shape != current_img.shape:
            prev_prev_img = cv2.resize(prev_prev_img, (current_img.shape[1], current_img.shape[0]))
        step_time = time.time() - start_time
        time_stats['image_resizing'] = step_time

        # 第五步：计算帧ID差值和比例因子
        start_time = time.time()
        delta_t1 = prev_frame_id - prev_prev_frame_id
        delta_t2 = current_frame_id - prev_frame_id
        scale_factor = delta_t2 / delta_t1 if delta_t1 != 0 else 1
        step_time = time.time() - start_time
        time_stats['frame_id_difference'] = step_time

        # 第六步：计算光流
        start_time = time.time()
        flow = self.compute_optical_flow_farneback(prev_prev_img, prev_img)
        step_time = time.time() - start_time
        time_stats['optical_flow'] = step_time

        # 第七步：使用光流预测当前帧
        start_time = time.time()
        predicted_img = self.warp_image(prev_img, flow, scale_factor)
        step_time = time.time() - start_time
        time_stats['image_prediction'] = step_time

        # 第八步：调整预测图像大小
        start_time = time.time()
        if predicted_img.shape != current_img.shape:
            predicted_img = cv2.resize(predicted_img, (current_img.shape[1], current_img.shape[0]))
        step_time = time.time() - start_time
        time_stats['predicted_image_resizing'] = step_time

        # 第九步：计算残差
        start_time = time.time()
        residual = cv2.absdiff(current_img, predicted_img)
        step_time = time.time() - start_time
        time_stats['residual_computation'] = step_time

        # 第十步：标记误差大于阈值的宏块
        start_time = time.time()
        # highlighted_img, mask, regions = self.highlight_large_errors(
        #     current_img, residual, threshold, block_size, current_frame_id, 8)
        highlighted_img, mask, regions = self.highlight_large_errors(
            current_img, residual, threshold, block_size,region_size, current_frame_id)
        step_time = time.time() - start_time
        time_stats['highlight_large_errors'] = step_time

        # 第十一步：添加区域结果
        start_time = time.time()
        # regions=Region.merge_regions(regions)
        for region in regions:
            region.resolution = current_img.shape[:2]
            results.add_single_result(region)
        step_time = time.time() - start_time
        time_stats['add_regions'] = step_time

        # 打印时间统计
        for step, duration in time_stats.items():
            print(f"{step}: {duration:.6f} seconds")

        return results

    #
    # def process_frames(self, current_img, current_frame_id, threshold=25, block_size=64):
    #     results = Results()
    #
    #     frame_ids = sorted(self.memory_cache.keys())
    #     if len(frame_ids) < 3:
    #         return results
    #
    #     # 取最大的和第二大的帧ID
    #     prev_frame_id = frame_ids[-1]
    #     prev_prev_frame_id = frame_ids[-2]
    #
    #     if prev_frame_id >= current_frame_id or prev_prev_frame_id >= current_frame_id:
    #         return results
    #
    #     prev_prev_img = self.ensure_grayscale(self.memory_cache[prev_prev_frame_id]["image"])
    #     prev_img = self.ensure_grayscale(self.memory_cache[prev_frame_id]["image"])
    #     current_img = self.ensure_grayscale(current_img)
    #
    #     # 调整图像大小以匹配
    #     if prev_img.shape != current_img.shape:
    #         prev_img = cv2.resize(prev_img, (current_img.shape[1], current_img.shape[0]))
    #     if prev_prev_img.shape != current_img.shape:
    #         prev_prev_img = cv2.resize(prev_prev_img, (current_img.shape[1], current_img.shape[0]))
    #
    #     # # 调试输出图像大小
    #     # print(f"Current Image Shape: {current_img.shape}")
    #     # print(f"Previous Image Shape: {prev_img.shape}")
    #     # print(f"Previous-Previous Image Shape: {prev_prev_img.shape}")
    #
    #     # 计算帧ID差值
    #     delta_t1 = prev_frame_id - prev_prev_frame_id
    #     delta_t2 = current_frame_id - prev_frame_id
    #     scale_factor = delta_t2 / delta_t1 if delta_t1 != 0 else 1
    #
    #     # 计算光流
    #     flow = self.compute_optical_flow_farneback(prev_prev_img, prev_img)
    #     # 使用光流预测当前帧
    #     predicted_img = self.warp_image(prev_img, flow, scale_factor)
    #
    #     # # 调试输出预测图像大小
    #     # print(f"Predicted Image Shape: {predicted_img.shape}")
    #
    #     # 调整预测图像大小以匹配
    #     if predicted_img.shape != current_img.shape:
    #         predicted_img = cv2.resize(predicted_img, (current_img.shape[1], current_img.shape[0]))
    #
    #     # # 调试输出调整后的预测图像大小
    #     # print(f"Resized Predicted Image Shape: {predicted_img.shape}")
    #
    #     # 计算残差
    #     residual = cv2.absdiff(current_img, predicted_img)
    #
    #     # 标记误差大于阈值的宏块
    #     highlighted_img, mask, regions = self.highlight_large_errors(current_img, residual, threshold, block_size,
    #                                                                  current_frame_id,8)
    #
    #     for region in regions:
    #         region.resolution = current_img.shape[:2]
    #         results.add_single_result(region)
    #
    #     return results
        # def process_frames(self, current_img, current_frame_id, threshold=10, block_size=32):
    #     current_img = self.ensure_grayscale(current_img)
    #     results = Results()
    #
    #     for prev_frame_id in sorted(self.memory_cache.keys()):
    #         if prev_frame_id >= current_frame_id:
    #             continue
    #
    #         prev_img = self.ensure_grayscale(self.memory_cache[prev_frame_id]["image"])
    #         if prev_img.shape != current_img.shape:
    #             prev_img = cv2.resize(prev_img, (current_img.shape[1], current_img.shape[0]))
    #
    #         # 计算光流
    #         flow = self.compute_optical_flow_farneback(prev_img, current_img)
    #         # 预测当前帧
    #         predicted_img = self.warp_image(prev_img, flow)
    #         # 计算残差
    #         residual = cv2.absdiff(current_img, predicted_img)
    #         # 标记误差大于阈值的宏块
    #         highlighted_img, mask, regions = self.highlight_large_errors(current_img, residual, threshold, block_size,
    #                                                                      current_frame_id)
    #
    #         for region in regions:
    #             region.resolution = current_img.shape[:2]
    #             results.add_single_result(region)
    #     return results
    def find_zero_motion_blocks_optimized(self,motion_vectors, block_size=16, zero_threshold=0.5):
        h, w, _ = motion_vectors.shape
        h_blocks = h // block_size
        w_blocks = w // block_size

        # 将 motion_vectors 重塑为 (h_blocks, block_size, w_blocks, block_size, 2)
        block_vectors = motion_vectors[:h_blocks * block_size, :w_blocks * block_size, :].reshape(h_blocks, block_size,
                                                                                                  w_blocks, block_size,
                                                                                                  2)

        # 计算块的平均运动向量
        block_means = np.mean(block_vectors, axis=(1, 3))

        # 计算每个块的运动向量的模
        magnitude = np.sqrt(block_means[..., 0] ** 2 + block_means[..., 1] ** 2)

        # 找到运动向量模小于阈值的块
        zero_motion_mask = magnitude < zero_threshold
        zero_motion_blocks = np.argwhere(zero_motion_mask)

        # 将块的索引转换为像素坐标
        zero_motion_blocks = [(x * block_size, y * block_size) for y, x in zero_motion_blocks]

        return zero_motion_blocks
    def find_background_blocks_ransac(self,motion_vectors, block_size=16, threshold=1.0, max_iters=1000):
        h, w, _ = motion_vectors.shape
        blocks = []
        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                block = motion_vectors[y:y + block_size, x:x + block_size, :]
                blocks.append(((x, y), block))

        points = []
        for (x, y), block in blocks:
            u = np.mean(block[..., 0])
            v = np.mean(block[..., 1])
            points.append([x + block_size / 2, y + block_size / 2, u, v])
        points = np.array(points)

        src_points = points[:, :2]
        dst_points = src_points + points[:, 2:]

        model, inliers = cv2.findHomography(src_points, dst_points, cv2.RANSAC, threshold)

        background_blocks = []
        for i, inlier in enumerate(inliers):
            if inlier:
                background_blocks.append(blocks[i][0])

        return background_blocks
    def extract_motion_vectors(self,image1, image2):
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        motion_vectors = np.dstack((flow[..., 0], flow[..., 1]))
        return motion_vectors

    def convert_results_to_df(self,results):
        """
        将 Results 对象转换为 DataFrame，包含 xmin, ymin, xmax, ymax 信息
        """
        data = []
        for region in results.regions:
            xmin = region.x
            ymin = region.y
            xmax = region.x + region.w
            ymax = region.y + region.h
            data.append({
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax,
                'conf': region.conf,
                'label': region.label
            })

        df = pd.DataFrame(data)
        return df

    def normalize_bbox_coordinates(self, df, original_width, original_height, new_width, new_height):
        # 检查DataFrame是否为空
        if df.empty:
            return df

        # 对xmin, ymin, xmax, ymax进行归一化
        df['xmin'] = df['xmin'] / original_width * new_width
        df['ymin'] = df['ymin'] / original_height * new_height
        df['xmax'] = df['xmax'] / original_width * new_width
        df['ymax'] = df['ymax'] / original_height * new_height

        return df
    def process_image_triplet(self,image1_path, image2_path, image3_path,results_df ,output_path=None):
        # Step 1: Start time for loading images
        start_time = time.time()

        image1 = cv2.imread(image1_path)
        image2 = cv2.imread(image2_path)
        image3 = cv2.imread(image3_path)

        if image1 is None or image2 is None or image3 is None:
            print(f"Error: One of the images could not be loaded.")
            return

        load_time = time.time() - start_time
        print(f"Time to load images: {load_time:.4f} seconds")

        # Step 2: Start time for processing images dimensions
        start_time = time.time()

        original_height, original_width, _ = image3.shape
        new_height, new_width, _ = image2.shape

        dimension_time = time.time() - start_time
        print(f"Time to process image dimensions: {dimension_time:.4f} seconds")

        # Step 3: Start time for object detection
        start_time = time.time()
        # 换成这句
        results_df = self.convert_results_to_df(results_df)
        # results_df = detect_objects(image3_path, conf_threshold=0.63, iou_threshold=0.1,
        #                             classes_to_detect=['person', 'car', 'bus', 'truck', 'bicycle', 'motorcycle'])
        results_df = self.normalize_bbox_coordinates(results_df, original_width, original_height, new_width, new_height)

        detection_time = time.time() - start_time
        print(f"Time for object detection and normalization: {detection_time:.4f} seconds")

        # Step 4: Start time for extracting motion vectors
        start_time = time.time()

        # 更快的方式，但不准
        # motion_vectors = extract_motion_vectors_with_scaling(image1, image2)

        motion_vectors = self.extract_motion_vectors(image1, image2)
        # motion_vectors = extract_motion_vectors_gpu(image1, image2)
        motion_vector_time = time.time() - start_time
        print(f"Time to extract motion vectors: {motion_vector_time:.4f} seconds")

        # Step 5: Start time for finding blocks
        start_time = time.time()

        h, w, _ = motion_vectors.shape
        all_blocks = [(x, y) for y in range(0, h, 16) for x in range(0, w, 16)]
        # background_blocks = find_background_blocks_ransac_optimized(motion_vectors)
        background_blocks = self.find_background_blocks_ransac(motion_vectors)
        # zero_motion_blocks = find_zero_motion_blocks(motion_vectors, zero_threshold=0.5)
        zero_motion_blocks = self.find_zero_motion_blocks_optimized(motion_vectors, block_size=16, zero_threshold=0.5)
        find_blocks_time = time.time() - start_time
        print(f"Time to find background and zero-motion blocks: {find_blocks_time:.4f} seconds")

        # Step 6: Start time for identifying bbox blocks and unmarked blocks
        start_time = time.time()

        block_size = 16
        bbox_blocks = set()
        for index, row in results_df.iterrows():
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            for y in range(y1, y2, block_size):
                for x in range(x1, x2, block_size):
                    bbox_blocks.add((x, y))

        marked_blocks = set(background_blocks + zero_motion_blocks + list(bbox_blocks))
        unmarked_blocks = [block for block in all_blocks if block not in marked_blocks]

        bbox_blocks_time = time.time() - start_time
        print(f"Time to identify bbox and unmarked blocks: {bbox_blocks_time:.4f} seconds")

        # Step 7: Start time for drawing rectangles
        # start_time = time.time()
        #
        # # for (x, y) in unmarked_blocks:
        # #     cv2.rectangle(image2, (x, y), (x + 16, y + 16), (0, 0, 255), 2)
        # #
        # # for (x, y) in bbox_blocks:
        # #     cv2.rectangle(image2, (x, y), (x + 16, y + 16), (0, 255, 0), 2)
        # #
        # # cv2.imwrite(output_path, image2)
        #
        # draw_time = time.time() - start_time
        # print(f"Time to draw rectangles and save the image: {draw_time:.4f} seconds")

        return all_blocks, unmarked_blocks, bbox_blocks



    def calculate_large_blocks(self, all_blocks_list, unmarked_blocks_list, bbox_blocks_list, large_block_width, large_block_height):
        """
        根据块列表计算需要填充的 large blocks。
        """
        # 使用字典记录哪些 large_block_width × large_block_height 的块需要填充
        large_blocks_to_fill = {}

        # 计算每个16×16块所属的 large_block_width × large_block_height 块并记录
        def mark_large_block(x, y):
            large_block_x = (x // large_block_width) * large_block_width
            large_block_y = (y // large_block_height) * large_block_height
            large_blocks_to_fill[(large_block_x, large_block_y)] = True

        # 处理BBOX和未标记块，将对应的 large_block_width × large_block_height 块标记为需要填充
        for (x, y) in bbox_blocks_list + unmarked_blocks_list:
            mark_large_block(x, y)

        return large_blocks_to_fill

    def normalize_coordinates(self,x, y, w, h, original_width, original_height):
            """
            对 Region 坐标进行归一化
            """
            x_norm = x / original_width
            y_norm = y / original_height
            w_norm = w / original_width
            h_norm = h / original_height
            return x_norm, y_norm, w_norm, h_norm

    def base_req_regions_res(self, high_images_path, start_fid, end_fid, results_df, large_block_width=64,
                             large_block_height=64, n=5,padding=1):
        """
        基于帧区间和大块尺寸，处理每个大块的感兴趣区域，并返回需要填充的 large blocks 和相应的 Region 对象。
        """
        # 如果 results_df 为空，返回默认 Region 对象
        if len(results_df.regions) == 0:
            base_req_regions_res = [Region(0, 0, 0, 1, 1, 1.0, 2, 1)]
            return base_req_regions_res, []
        # 确定要选择的帧ID
        selected_fids = np.linspace(start_fid, end_fid - 1, n, dtype=int)

        # 存储所有块的信息
        all_blocks_list = []
        unmarked_blocks_list = []
        bbox_blocks_list = []

        def process_single_triplet(idx, fid):
            # 使用fid读取对应的三张连续帧图像
            image1_path = os.path.join(high_images_path, f"{fid:08d}.jpg")
            image2_path = os.path.join(high_images_path, f"{fid + 1:08d}.jpg")
            image3_path = os.path.join(high_images_path, f"{fid + 2:08d}.jpg")

            # 确保所有图像存在
            if not os.path.exists(image1_path) or not os.path.exists(image2_path) or not os.path.exists(image3_path):
                print(f"Image(s) missing for fid: {fid}")
                # 即使图像缺失，依然返回 5 个值，默认宽高为 None
                return [], [], [], None, None

            # 读取图像并获取宽度和高度
            img = cv2.imread(image1_path)
            if img is None:
                return [], [], [], None, None

            original_height, original_width = img.shape[:2]

            # 从 process_image_triplet 函数获取 (x, y, w, h) 形式的元组
            all_blocks, unmarked_blocks, bbox_blocks = self.process_image_triplet(image1_path, image2_path, image3_path,
                                                                                  results_df)

            # 确保返回 5 个值
            return all_blocks, unmarked_blocks, bbox_blocks, original_width, original_height

        # 多线程处理每批次
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(process_single_triplet, range(len(selected_fids)), selected_fids))

        # 收集所有的块
        original_width = original_height = None
        # 遍历 results，处理 all_blocks, unmarked_blocks, bbox_blocks 和图像宽高
        for all_blocks, unmarked_blocks, bbox_blocks, img_width, img_height in results:
            # 如果当前元组的任何部分是空的，直接跳过
            if img_width is None or img_height is None:
                continue

            # 初始化图像的宽度和高度
            if original_width is None or original_height is None:
                original_width, original_height = img_width, img_height

            # 扩展块列表
            all_blocks_list.extend(all_blocks)
            unmarked_blocks_list.extend(unmarked_blocks)
            bbox_blocks_list.extend(bbox_blocks)

        # 计算 large_blocks_to_fill
        large_blocks_to_fill = self.calculate_large_blocks(all_blocks_list, unmarked_blocks_list, bbox_blocks_list,
                                                           large_block_width, large_block_height)

        # 基于 large_blocks_to_fill 生成 Region 对象
        base_req_regions_res = []
        for (large_block_x, large_block_y) in large_blocks_to_fill:
            # 对 block 的坐标进行归一化，并添加 padding
            x_norm, y_norm, w_norm, h_norm = self.normalize_coordinates(
                large_block_x - padding, large_block_y - padding,
                large_block_width + 2 * padding, large_block_height + 2 * padding,
                original_width=original_width, original_height=original_height
            )

            # 创建归一化后的 Region 对象
            base_req_regions_res.append(
                Region(selected_fids[0], x_norm, y_norm, w_norm, h_norm, 1.0, 2, 1)
            )

        return base_req_regions_res, large_blocks_to_fill


