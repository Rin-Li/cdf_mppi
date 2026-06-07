import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="CDF-MPPI: Motion Planning with Configuration Space Distance Fields",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--method",
        choices=["cdf_mppi", "mppi", "qp"],
        default="cdf_mppi",
        help=(
            "Planning method:\n"
            "  cdf_mppi  CDF-MPPI (ours, H=1, recommended)\n"
            "  mppi      Original MPPI\n"
            "  qp        QP/IPOPT baseline"
        ),
    )
    parser.add_argument(
        "--robot",
        choices=["2d", "franka"],
        default="2d",
        help=(
            "Robot:\n"
            "  2d      2-link planar robot (no extra deps)\n"
            "  franka  7-DOF Franka Panda (requires pybullet)"
        ),
    )
    args = parser.parse_args()

    if args.robot == "2d":
        if args.method == "cdf_mppi":
            from exp.linkrobot.cdf_mppi import main as run
        elif args.method == "mppi":
            from exp.linkrobot.mppi_original import main as run
        elif args.method == "qp":
            from exp.linkrobot.qp_2d import main as run
    elif args.robot == "franka":
        if args.method != "cdf_mppi":
            print("Only cdf_mppi is supported for the franka robot in this demo.")
            sys.exit(1)
        from exp.franka.cdf_mppi import main as run

    run()


if __name__ == "__main__":
    main()
