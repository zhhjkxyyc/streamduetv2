import os
import time
import pickle
from collections import deque
import threading
import cv2
import numpy as np
from filelock import FileLock

class InferenceCache:
    def __init__(self, time_window, cache_dir, conf_threshold, update_interval=0.1):
        self.time_window = time_window  # Define the time window for frame search
        self.cache_dir = cache_dir
        self.conf_threshold = conf_threshold  # Confidence threshold for results
        self.memory_cache = {}
        self.update_interval = update_interval
        self._stop_event = threading.Event()
        self.cache_lock = threading.Lock()
        # Start the cache update thread
        self.cache_update_thread = threading.Thread(target=self.update_cache_periodically)
        self.cache_update_thread.daemon = True  # Daemon thread will exit with the program
        self.cache_update_thread.start()
    def update_cache_periodically(self):
        while not self._stop_event.is_set():
            print("Updating cache from disk...")
            self.update_memory_cache()  # Load from cache.pkl
            time.sleep(self.update_interval)


    def update_memory_cache(self):
        """
        This function will load the cache from the cache.pkl file to the memory cache.
        Runs in the background thread.
        """
        while not self._stop_event.is_set():
            # Update memory cache from disk (cache.pkl)
            for root, _, files in os.walk(self.cache_dir):
                for file in files:
                    if file == "cache.pkl":
                        cache = self._load_cache(os.path.join(root, file))
                        # Acquire lock to update the cache safely
                        with self.cache_lock:
                            for frame_id, timestamp, results, image_path in cache:
                                # Add or update the memory cache
                                self.memory_cache[image_path] = (frame_id, results)
            time.sleep(self.update_interval)  # Wait before the next update

    def _current_time(self):
        return time.time()

    def _get_cache_path(self, video_name):
        return os.path.join(self.cache_dir, video_name, "cache.pkl")

    def _get_image_path(self, video_name, frame_id):
        return os.path.join(self.cache_dir, video_name, f"{frame_id}.jpg")

    def _get_lock_path(self, video_name):
        return os.path.join(self.cache_dir, video_name, "cache.lock")

    def _load_cache(self, path):
        if not os.path.exists(path):
            return deque()
        try:
            with open(path, 'rb') as f:
                cache = pickle.load(f)
        except (EOFError, pickle.UnpicklingError):
            return deque()
        # 恢复 keypoints 对象
        for frame_id, timestamp, results, image_path in cache:
            for region in results:
                if region.feature is not None and 'keypoints' in region.feature:
                    region.feature['keypoints'] = self.dict_to_keypoints(region.feature['keypoints'])
        return cache
    def stop(self):
        self._stop_event.set()
        self.cache_update_thread.join()
    def _save_cache(self, path, cache):
        # 将 keypoints 转换为可序列化的格式
        for frame_id, timestamp, results, image_path in cache:
            for region in results:
                if region.feature is not None and 'keypoints' in region.feature:
                    region.feature['keypoints'] = self.keypoints_to_dict(region.feature['keypoints'])
        with open(path, 'wb') as f:
            pickle.dump(cache, f)
        # 恢复 keypoints 对象以便继续使用
        for frame_id, timestamp, results, image_path in cache:
            for region in results:
                if region.feature is not None and 'keypoints' in region.feature:
                    region.feature['keypoints'] = self.dict_to_keypoints(region.feature['keypoints'])

    def keypoints_to_dict(self, keypoints):
        return [(kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in keypoints]

    def dict_to_keypoints(self, kp_dict):
        keypoints = []
        for pt, size, angle, response, octave, class_id in kp_dict:
            keypoints.append(
                cv2.KeyPoint(pt[0], pt[1], size, angle, response, octave, class_id))
        return keypoints
    def add_results(self, video_name, frame_id, results, frame_image):
        video_cache_dir = os.path.join(self.cache_dir, video_name)
        os.makedirs(video_cache_dir, exist_ok=True)
        cache_path = self._get_cache_path(video_name)
        image_path = self._get_image_path(video_name, frame_id)

        # Use lock to ensure thread-safe cache modification
        with self.cache_lock:
            # Check if the frame image is in memory cache or disk
            # if image_path not in self.memory_cache and not os.path.exists(image_path):
                # cv2.imwrite(image_path, frame_image)
                # Save the frame image to memory cache
                # self.memory_cache[image_path] = frame_image

            # Load, update, and save the cache (both in-memory and on-disk)
            cache = self._load_cache(cache_path)

            # For each region in the results, extract features if needed
            for region in results:
                if region.feature is None:
                    cropped_cached_image = self._crop_image(frame_image, region)
                    feature = self.extract_features(cropped_cached_image)
                    region.feature = feature

            # Append the new result to the cache
            cache.append((frame_id, self._current_time(), results, image_path))

            # Clean up cache if necessary and save to disk
            self._cleanup()
            self._save_cache(cache_path, cache)

            # Also update the memory cache with the latest results
            self.memory_cache[image_path] = (frame_id, results)

    def get_best_result(self, bbox, query_image):
        best_result = None
        best_image_path = None

        # Access the in-memory cache (use lock to ensure safe access)
        with self.cache_lock:
            for image_path, (frame_id, results) in self.memory_cache.items():
                current_frame = self._get_max_frame(self.memory_cache)
                min_frame = current_frame - self.time_window

                # Only process results within the time window
                if frame_id < min_frame:
                    continue

                cropped_query_image = self._crop_image(query_image, bbox)

                for region in results:
                    # Match the feature if the confidence threshold is met
                    if region.conf >= self.conf_threshold:
                        if self.match_features(self.extract_features(cropped_query_image), region.feature):
                            if best_result is None or region.conf > best_result.conf:
                                best_result = region
                                best_image_path = image_path
                                bbox.conf = best_result.conf
                                bbox.label = best_result.label
                                bbox.origin = "low-cache-res"
                                bbox.feature = region.feature
                                return bbox, best_image_path

        return best_result, best_image_path

    def _cleanup(self):
        if not self.memory_cache:
            return

        # Find the maximum frame ID currently in memory cache
        current_frame = self._get_max_frame(self.memory_cache)

        # Calculate the minimum valid frame based on the time window
        min_frame = current_frame - self.time_window

        # Create a list of image paths to remove from memory_cache
        keys_to_remove = []

        # Traverse through the memory cache
        for image_path, (frame_id, results) in self.memory_cache.items():
            # If frame_id is older than the time window, mark it for removal
            if frame_id < min_frame:
                keys_to_remove.append(image_path)

        # Remove the old frames from memory_cache and disk
        for image_path in keys_to_remove:
            # Remove from memory cache
            del self.memory_cache[image_path]
            # Optionally remove the corresponding image file from the disk if it exists
            if os.path.exists(image_path):
                os.remove(image_path)

    def _get_max_frame(self, cache):
        if not cache:
            return -1

        valid_frame_ids = []
        invalid_frame_ids = []

        for video_path, frame_data_list in cache.items():
            # 直接检查是否是整数，减少异常处理
            frame_id = frame_data_list[0]
            if isinstance(frame_id, int):
                valid_frame_ids.append(frame_id)
            else:
                invalid_frame_ids.append(frame_id)

        # 使用条件运算符返回结果
        return max(valid_frame_ids, default=-1)

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

        # Apply ratio test
        good_matches = []
        for match in matches:
            if len(match) == 2:
                m, n = match
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        return len(good_matches) > 10  # Threshold for considering a match

    def match_features(self,query_features, cached_features, ratio_threshold=0.75, match_threshold=10):
        query_keypoints, query_descriptors = query_features['keypoints'], query_features['descriptors']
        cached_keypoints, cached_descriptors = cached_features['keypoints'], cached_features['descriptors']

        # 检查描述符是否为空
        if query_descriptors is None or cached_descriptors is None:
            return False

        # 创建BFMatcher对象
        bf = cv2.BFMatcher()

        # 使用KNN算法进行特征匹配
        matches = bf.knnMatch(query_descriptors, cached_descriptors, k=2)

        # 应用比例测试来过滤匹配
        good_matches = []
        for match in matches:
            if len(match) == 2:
                m, n = match
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)

        # 判断匹配的数量是否超过阈值
        return len(good_matches) > match_threshold
    def extract_features(self,image):
        # 初始化SIFT检测器
        sift = cv2.SIFT_create()

        # 检测并计算特征点和描述符
        keypoints, descriptors = sift.detectAndCompute(image, None)

        return {'keypoints': keypoints, 'descriptors': descriptors}

# class BBox:
#     def __init__(self, x, y, w, h, conf, label):
#         self.x = x
#         self.y = y
#         self.w = w
#         self.h = h
#         self.conf = conf
#         self.label = label
