import argparse
def parameter_parser():
    parser = argparse.ArgumentParser(description = "Run SEAL-CI/SEAL-AI.")

    parser.add_argument("--graphs",
                        nargs="?",
                        default="/OpenWGL-main/opgl/dataset/test1/",
                        help="Folder with training graph jsons.")

    parser.add_argument("--hierarchical-graph",
                        nargs="?",
                        default="/OpenWGL-main/opgl/dataset/test1/edges_DoS.csv",
                        help="Hierarchical edge list.")

    parser.add_argument("--labeled-count",
                        type=int,
                        default=1000,
                        help="Number of labeled data points. Default is 100.")
    parser.add_argument("--first-gcn-dimensions",
                        type=int,
                        default=64,
                        help="Filters (neurons) in 1st convolution. Default is 32.")

    parser.add_argument("--second-gcn-dimensions",
                        type=int,
                        default=32,
                        help="Filters (neurons) in 2nd convolution. Default is 16.")

    parser.add_argument("--first-dense-neurons",
                        type=int,
                        default=32,
                        help="Neurons in SAGE aggregator layer. Default is 16.")

    parser.add_argument("--second-dense-neurons",
                        type=int,
                        default=16,
                        help="SAGE attention neurons. Default is 8.")
    parser.add_argument("--feature_which",
                        type=list,
                        default=[0, 1, 2, 3, 4, 5],
                        help="Attention regularization coefficient. Default is 10^-5.")
    parser.add_argument("--unseen_num",
                        type=list,
                        default=1,
                        help="Attention regularization coefficient. Default is 10^-5.")
    parser.add_argument("--gamma",
                        type=float,
                        default=10**-5
                        )
    return parser.parse_args()
