import yaml
import sys
import os
import subprocess
from copy import deepcopy
import threading

sys.path.append('../')
from instance_executor import InstanceFactory

def load_configuration():
    with open('configuration.yml', 'r') as config:
        config_info = yaml.safe_load(config)
    return config_info

def parameter_sweeping(instances, new_instance, keys, config_info,config_dict, data_dir):
    if not keys:
        threads = []
        for video_name in config_info['video_names']:
            instance_config = config_dict[video_name]
            executor = InstanceFactory.get_instance_executor(new_instance, video_name, instance_config, data_dir)
            t = threading.Thread(target=executor.execute, args=())
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
    else:
        curr_key = keys[0]
        if isinstance(instances[curr_key], list):
            for each_parameter in instances[curr_key]:
                new_instance[curr_key] = each_parameter
                parameter_sweeping(instances, new_instance, keys[1:], config_info,config_dict, data_dir)
        else:
            new_instance[curr_key] = instances[curr_key]
            parameter_sweeping(instances, new_instance, keys[1:], config_info,config_dict, data_dir)

def create_instance_config_dict(config_info):
    print("create_instance_config_dict is defined and called")  # Add this line
    instance_config_dict = {}
    for video_name in config_info['video_names']:
        instance_config_info = deepcopy(config_info)
        instance_config_info['video_name'] = video_name
        instance_config_dict[video_name] = instance_config_info
    return instance_config_dict

def execute_instance(instance, video_name, config_info, data_dir):
    new_instance = {}
    for key in config_info['default'].keys():
        if key not in instance.keys():
            instance[key] = config_info['default'][key]

    print("Calling create_instance_config_dict")  # Add this line
    config_dict = create_instance_config_dict(config_info)

    parameter_sweeping(instance, new_instance, list(instance.keys()), config_info, config_dict, data_dir)

def execute_all(config_info, data_dir):
    all_instances = config_info['instances']
    for instance in all_instances:
        execute_instance(instance, None, config_info, data_dir)

if __name__ == "__main__":
    config_info = load_configuration()
    data_dir = config_info['data_dir']
    print("Starting all executions")
    execute_all(config_info, data_dir)
    print("All executions completed")
