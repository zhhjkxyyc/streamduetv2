import os
import shutil
import cv2 as cv
from dds_utils import (Results, extract_images_from_video, Region)
from streamduet_utils import list_frames,get_images_length,get_image_extension
def perform_server_cleanup():
    for f in os.listdir("server_temp"):
        os.remove(os.path.join("server_temp", f))
    for f in os.listdir("server_temp-cropped"):
        os.remove(os.path.join("server_temp-cropped", f))

def reset_server_state(nframes):
    for f in os.listdir("server_temp"):
        os.remove(os.path.join("server_temp", f))
    for f in os.listdir("server_temp-cropped"):
        os.remove(os.path.join("server_temp-cropped", f))

def perform_detection(images_direc, resolution, fnames, images, detector, config, logger):
    final_results = Results()
    rpn_regions = Results()
    image_extension=get_image_extension(images_direc)
    if fnames is None:
        fnames = sorted(os.listdir(images_direc))
    logger.info(f"Running inference on {len(fnames)} frames")
    for fname in fnames:
        if image_extension not in fname:
            continue
        fid = int(fname.split(".")[0])
        if fid in images:
            image = images[fid]
        else:
            image_path = os.path.join(images_direc, fname)
            image = cv.imread(image_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        detection_results, rpn_results = detector.infer(image)
        frame_with_no_results = True
        for label, conf, (x, y, w, h) in detection_results:
            if (config.min_object_size and w * h < config.min_object_size) or w * h == 0.0:
                continue
            r = Region(fid, x, y, w, h, conf, label, resolution, origin="mpeg")
            final_results.append(r)
            frame_with_no_results = False
        for label, conf, (x, y, w, h) in rpn_results:
            r = Region(fid, x, y, w, h, conf, label, resolution, origin="generic")
            rpn_regions.append(r)
            frame_with_no_results = False
        logger.debug(f"Got {len(final_results)} results and {len(rpn_regions)} for {fname}")

        if frame_with_no_results:
            final_results.append(Region(fid, 0, 0, 0, 0, 0.1, "no obj", resolution))

    return final_results, rpn_regions

def extract_images(vid_data):
    with open(os.path.join("server_temp", "temp.mp4"), "wb") as f:
        f.write(vid_data.read())