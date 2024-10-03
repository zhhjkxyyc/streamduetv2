import os
import subprocess
import yaml
import uuid
from abc import ABC, abstractmethod
from copy import deepcopy

class InstanceExecutor(ABC):
    def __init__(self, instance, video_name, config_info, data_dir):
        self.instance = instance
        self.video_name = video_name
        self.config_info = config_info
        self.client_id = str(uuid.uuid4())
        self.instance['client_id'] = self.client_id
        self.instance['video_name'] = self.video_name
        model_type = self.instance['model_type']
        self.instance['model'] = self.config_info['models'][model_type]
        self.instance['relevant_classes'] = self.config_info['default']['relevant_classes']
        self.data_dir = data_dir
    @abstractmethod
    def execute(self):
        pass

class GTInstance(InstanceExecutor):
    def execute(self):
        result_file_name = f"{self.video_name}_gt"
        if not self.instance['overwrite'] and os.path.exists(os.path.join("results", result_file_name)):
            print(f"Skipping {result_file_name}")
        else:
            self.instance['video_name'] = f'results/{result_file_name}'
            self.instance['high_images_path'] = f'{os.path.join(self.data_dir, self.video_name)}'
            self.instance['outfile'] = 'stats'
            subprocess.run(['python', '../play_video.py', yaml.dump(self.instance)])

class MPEGInstance(InstanceExecutor):
    def execute(self):
        result_file_name = f"{self.video_name}_mpeg_{self.instance['low_resolution']}_{self.instance['low_qp']}"
        if not self.instance['overwrite'] and os.path.exists(os.path.join("results", result_file_name)):
            print(f"Skipping {result_file_name}")
        else:
            self.instance['video_name'] = f'results/{result_file_name}'
            self.instance['high_images_path'] = f'{os.path.join(self.data_dir, self.video_name)}'
            self.instance['outfile'] = 'stats'
            self.instance['ground_truth'] = f'results/{self.video_name}_gt'
            subprocess.run(['python', '../play_video.py', yaml.dump(self.instance)])

class DDSInstance(InstanceExecutor):
    def execute(self):
        result_file_name = (
            f"{self.video_name}_dds_{self.instance['low_resolution']}_{self.instance['high_resolution']}_"
            f"{self.instance['low_qp']}_{self.instance['high_qp']}_{self.instance['rpn_enlarge_ratio']}_"
            f"twosides_batch_{self.instance['batch_size']}_{self.instance['prune_score']}_"
            f"{self.instance['objfilter_iou']}_{self.instance['size_obj']}"
        )
        if not self.instance['overwrite'] and os.path.exists(os.path.join("results", result_file_name)):
            print(f"Skipping {result_file_name}")
        else:
            self.instance['video_name'] = f'results/{result_file_name}'
            self.instance['high_images_path'] = f'{os.path.join(self.data_dir, self.video_name)}'
            self.instance['outfile'] = 'stats'
            self.instance['ground_truth'] = f'results/{self.video_name}_gt'
            self.instance['low_results_path'] = f"results/{self.video_name}_mpeg_{self.instance['low_resolution']}_{self.instance['low_qp']}"
            subprocess.run(['python', '../play_video.py', yaml.dump(self.instance)])

class StreamDuetInstance(InstanceExecutor):
    def execute(self):
        result_file_name = (
            f"{self.video_name}_streamduet_{self.instance['low_resolution']}_{self.instance['high_resolution']}_"
            f"{self.instance['low_qp']}_{self.instance['high_qp']}_{self.instance['rpn_enlarge_ratio']}_"
            f"twosides_batch_{self.instance['batch_size']}_{self.instance['prune_score']}_"
            f"{self.instance['objfilter_iou']}_{self.instance['size_obj']}"
        )
        if not self.instance['overwrite'] and os.path.exists(os.path.join("results", result_file_name)):
            print(f"Skipping {result_file_name}")
        else:
            self.instance['video_name'] = f'results/{result_file_name}'
            self.instance['high_images_path'] = f'{os.path.join(self.data_dir, self.video_name)}'
            self.instance['outfile'] = 'stats'
            self.instance['ground_truth'] = f'results/{self.video_name}_gt'
            self.instance['low_results_path'] = f"results/{self.video_name}_mpeg_{self.instance['low_resolution']}_{self.instance['low_qp']}"
            if self.instance["mode"] == 'implementation':
                assert self.instance['hname'] != False, "Must provide the server address for implementation, abort."
                # single_instance['hname'] = '127.0.0.1:5000'
            subprocess.run(['python', '../play_video.py', yaml.dump(self.instance)])


class StreamDuetRoIInstance(InstanceExecutor):
    def execute(self):
        result_file_name = (
            f"{self.video_name}_streamduetRoI_{self.instance['low_resolution']}_{self.instance['high_resolution']}_"
            f"{self.instance['low_qp']}_{self.instance['high_qp']}_{self.instance['rpn_enlarge_ratio']}_"
            f"twosides_batch_{self.instance['batch_size']}_{self.instance['prune_score']}_"
            f"{self.instance['objfilter_iou']}_{self.instance['size_obj']}"
        )
        if not self.instance['overwrite'] and os.path.exists(os.path.join("results", result_file_name)):
            print(f"Skipping {result_file_name}")
        else:
            self.instance['video_name'] = f'results/{result_file_name}'
            self.instance['high_images_path'] = f'{os.path.join(self.data_dir, self.video_name)}'
            self.instance['outfile'] = 'stats'
            self.instance['ground_truth'] = f'results/{self.video_name}_gt'
            self.instance['low_results_path'] = f"results/{self.video_name}_mpeg_{self.instance['low_resolution']}_{self.instance['low_qp']}"
            if self.instance["mode"] == 'implementation':
                assert self.instance['hname'] != False, "Must provide the server address for implementation, abort."
                # single_instance['hname'] = '127.0.0.1:5000'
            subprocess.run(['python', '../play_video.py', yaml.dump(self.instance)])

class InstanceFactory:
    @staticmethod
    def get_instance_executor(instance, video_name, instance_config, data_dir):
        # 添加server参数传递
        if instance['method'] == 'gt':
            return GTInstance(instance, video_name, instance_config, data_dir)
        elif instance['method'] == 'mpeg':
            return MPEGInstance(instance, video_name, instance_config, data_dir)
        elif instance['method'] == 'dds':
            return DDSInstance(instance, video_name, instance_config, data_dir)
        elif instance['method'] == 'streamduet':
            return StreamDuetInstance(instance, video_name, instance_config, data_dir)
        elif instance['method'] == 'streamduetRoI':
            return StreamDuetRoIInstance(instance, video_name, instance_config, data_dir)
        else:
            raise ValueError(f"Unknown method {instance_config['method']}")