import re
import logging
import sys
import os
from munch import munchify
import yaml
from dds_utils import read_results_dict, evaluate, write_stats
from workspace.instance_strategy import StrategyFactory  # 确保导入路径正确
from backend.server import Server


def main(args):
    logging.basicConfig(
        format="%(name)s -- %(levelname)s -- %(lineno)d -- %(message)s",
        level=args.verbosity.upper())

    logger = logging.getLogger("streamduet")
    logger.addHandler(logging.NullHandler())

    logger.info(
        f"Starting server with high threshold of {args.high_threshold} low threshold of {args.low_threshold} tracker length of {args.tracker_length}")

    config = args
    strategy = StrategyFactory.get_strategy(args, config, logger)
    results, bw = strategy.run(args)

    mode = args.method  # 使用 args.method 作为 mode

    # Evaluation and writing results
    low, high = bw
    f1 = 0
    stats = (0, 0, 0)
    number_of_frames = len([x for x in os.listdir(args.high_images_path) if "jpg" in x])
    if args.ground_truth:
        ground_truth_dict = read_results_dict(args.ground_truth)
        logger.info("Reading ground truth results complete")
        tp, fp, fn, _, _, _, f1 = evaluate(number_of_frames - 1, results.regions_dict, ground_truth_dict,
                                           args.low_threshold, 0.5, 0.4, 0.4)
        stats = (tp, fp, fn)
        logger.info(
            f"Got an f1 score of {f1} for this experiment {mode} with tp {stats[0]} fp {stats[1]} fn {stats[2]} with total bandwidth {sum(bw)}")
    else:
        logger.info("No groundtruth given skipping evaluation")

    # Write evaluation results to file
    write_stats(args.outfile, f"{args.video_name}", config, f1, stats, bw, number_of_frames, mode)


if __name__ == "__main__":
    args = munchify(yaml.load(sys.argv[1], Loader=yaml.SafeLoader))

    if not args.simulate and not args.hname and args.high_resolution != -1:
        if not args.high_images_path:
            print("Running DDS in emulation mode requires raw/high resolution images")
            exit()

    if not re.match("DEBUG|INFO|WARNING|CRITICAL", args.verbosity.upper()):
        print(
            "Incorrect argument for verbosity. Verbosity can only be one of the following:\n\tdebug\n\tinfo\n\twarning\n\terror")
        exit()

    if args.estimate_banwidth and not args.high_images_path:
        print("DDS needs location of high resolution images to calculate true bandwidth estimate")
        exit()

    if not args.simulate and args.high_resolution != -1:
        if args.low_images_path:
            print("Discarding low images path")
            args.low_images_path = None
        args.intersection_threshold = 1.0

    if not (args.method == "dds" or args.method == "streamduet" or args.method == "streamduetRoI"):
        assert args.high_resolution == -1, "Only dds and streamduet support two quality levels"

    if args.high_resolution == -1:
        print("Only one resolution given, running MPEG emulation")
        assert args.high_qp == -1, "MPEG emulation only support one QP"
    else:
        assert args.low_resolution <= args.high_resolution, f"The resolution of low quality({args.low_resolution}) can't be larger than high quality({args.high_resolution})"
        assert not (
                    args.low_resolution == args.high_resolution and args.low_qp < args.high_qp), f"Under the same resolution, the QP of low quality({args.low_qp}) can't be higher than the QP of high quality({args.high_qp})"

    main(args)
