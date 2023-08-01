"""Entry point for running GoTube."""
import configparser
import os
import numpy as np
import jax.numpy as jnp

import gotube.benchmarks as bm
import gotube.stochastic_reachtube as reach
import gotube.go_tube as go_tube
import time
from gotube.performance_log import close_log
from gotube.performance_log import create_plot_file
from gotube.performance_log import write_plot_file
from gotube.performance_log import log_stat


def run_gotube(system: bm.BaseSystem, args):
    """Run GoTube for the given system

    Parameters
    ----------
    system : bm.BaseSystem
        The system to be verified
    args : _type_
        Command line arguments
    """
    start_time = time.time()
    config = configparser.ConfigParser()
    config.read(os.path.dirname(__file__) + "/config.ini")
    files = config["files"]
    rt = reach.StochasticReachtube(
        system=bm.get_model(system, args.radius),
        profile=args.profile,
        mu=args.mu,  # mu as maximum over-approximation
        gamma=args.gamma,  # error-probability
        batch=args.batch_size,
        num_gpus=args.num_gpus,
        fixed_seed=args.fixed_seed,
        radius=args.radius,
    )  # reachtube

    timeRange = np.arange(args.starting_time + args.time_step, args.time_horizon + 1e-9, args.time_step)
    timeRange_with_start = np.append(np.array([0]), timeRange)

    volume = jnp.array([rt.compute_volume()])

    # propagate center point and compute metric
    (
        cx_timeRange,
        A1_timeRange,
        M1_timeRange,
        semiAxes_prod_timeRange,
    ) = rt.compute_metric_and_center(
        timeRange_with_start,
        args.ellipsoids,
    )

    fname = create_plot_file(files)

    write_plot_file(fname, "w", 0, rt.model.cx, rt.model.rad, M1_timeRange[0, :, :])

    total_random_points = None
    total_gradients = None
    total_initial_points = jnp.zeros((rt.num_gpus, 0, rt.model.dim))

    # for loop starting at Line 2
    for i, time_py in enumerate(timeRange):
        print(f"Step {i}/{timeRange.shape[0]} at {time_py:0.2f} s")

        rt.time_horizon = time_py
        rt.time_step = args.time_step
        rt.cur_time = rt.time_horizon

        rt.cur_cx = cx_timeRange[i + 1, :]
        rt.A1 = A1_timeRange[i + 1, :, :]

        #  GoTube Algorithm for t = t_j
        #  while loop from Line 5 - Line 15
        (
            rt.cur_rad,
            prob,
            total_initial_points,
            total_random_points,
            total_gradients,
        ) = go_tube.optimize(
            rt, total_initial_points, total_random_points, total_gradients
        )

        write_plot_file(
            fname, "a", time_py, rt.cur_cx, rt.cur_rad, M1_timeRange[i + 1, :, :]
        )

        volume = jnp.append(volume, rt.compute_volume(semiAxes_prod_timeRange[i + 1]))

        if rt.profile:
            # If profiling is enabled, log some statistics about the GD optimization process
            volumes = {
                "volume": float(volume[-1]),
                "average_volume": float(volume.mean()),
            }
            log_stat(volumes)

    if args.score:
        with open("all_prob_scores.csv", "a") as f:
            # CSV with header benchmark, time-horizon, prob, runtime, volume
            f.write(f"{args.benchmark},")
            f.write(f"{args.time_horizon:0.4g},")
            f.write(f"{args.radius:0.4g},")
            f.write(f"{args.mu:0.4g},")
            f.write(f"{1.0-args.gamma:0.4f},")
            f.write(f"{time.time()-start_time:0.2f},")
            f.write(f"{total_random_points.shape[0]:d},")
            f.write(f"{float(volume.mean()):0.5g}")
            f.write("\n")
    if rt.profile:
        final_notes = {
            "total_time": time.time() - start_time,
            "samples": args.num_gpus * total_random_points.shape[1],
        }
        close_log(final_notes)
