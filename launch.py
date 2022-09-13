from launch_utils import get_argument_parser, get_experience_streams, get_model_config_dict, set_seeds, get_log_param_dict, create_log_dirs
from NISPA import architecture, learner
import os


if __name__ == '__main__':
    args = get_argument_parser()
    log_params = get_log_param_dict(args)
    create_log_dirs(args, log_params)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if args.deterministic: set_seeds(args.seed)
    scenario, input_size, output_size = get_experience_streams(args)
    config_dict = get_model_config_dict(args)
    network = architecture.Network(config_dict, input_size, output_size, args)
    nispa = learner.Learner(args, network, scenario, log_params)
    network = nispa.learn_tasks()
