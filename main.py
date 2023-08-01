import argparse
from jax import config
from gotube.performance_log import log_args
from gotube.run import run_gotube
config.update("jax_enable_x64", True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--benchmark", default="mlp")
    # starting_time, time_step and time_horizon for creating reachtubes
    parser.add_argument("--starting_time", default=0.0, type=float)
    parser.add_argument("--time_step", default=0.01, type=float)
    parser.add_argument("--time_horizon", default=10, type=float)
    # batch-size for tensorization
    parser.add_argument("--batch_size", default=10000, type=int)
    # number of GPUs for parallelization
    parser.add_argument("--num_gpus", default=1, type=int)
    # use fixed seed for random points (only for comparing different algorithms)
    parser.add_argument("--fixed_seed", action="store_true")
    # error-probability
    parser.add_argument("--gamma", default=0.2, type=float)
    # mu as maximum over-approximation
    parser.add_argument("--mu", default=1.5, type=float)
    # choose between hyperspheres and ellipsoids to describe the Reachsets
    parser.add_argument("--ellipsoids", action="store_true")
    # initial radius
    parser.add_argument("--radius", default=None, type=float)

    args = parser.parse_args()
    log_args(vars(args))
    run_gotube(args.benchmark, args)
