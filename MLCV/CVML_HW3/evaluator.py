
import argparse
import time
from typing import Callable
import hw3

parser = argparse.ArgumentParser()
parser.add_argument("--train_data", default="segmentation_data.mat", type=str, help="The training data file path.")
parser.add_argument("--test_data", default=None, type=str, help="The testing data file path.")
parser.add_argument("--feature", default=False, action="store_true", help="Runs the feature analysis.")
parser.add_argument("--cluster", default=False, action="store_true", help="Runs the cluster analysis.")
parser.add_argument("--train_search", default=False, action="store_true", help="Runs the train search for best parameters.")
parser.add_argument("--train_best", default=False, action="store_true", help="Trains the models with the best parameters.")
parser.add_argument("--evaluate", default=False, action="store_true", help="Runs the evaluation on test data.")
parser.add_argument("--split_test", default=False, action="store_true", help="Whether the training data should be split for evaluation.")
parser.add_argument("--seed", default=42, type=int, help="Seed for RNG.")

def measureTime(func : Callable[[], None]) -> float:
    start = time.time()
    func()
    end = time.time()
    return end - start

def main(args : argparse.Namespace) -> None:
    print("Running the data analysis of image segmentation.")
    print("- Training data: {}".format(args.train_data))
    print("- Testing data:  {}".format(args.test_data))

    data_analysis = hw3.DataAnalysis(args.train_data, args.split_test, args.seed)
    if args.feature:
        print("Running the feature analysis.")
        elapsed = measureTime(lambda : data_analysis.featureAnalysis())
        print("Finished feature analysis in {:.4f} seconds.".format(elapsed))

    if args.cluster:
        print("Running the cluster analysis.")
        elapsed = measureTime(lambda : data_analysis.clusterAnalysis())
        print("Finished cluster analysis in {:.4f} seconds.".format(elapsed))

    if args.train_search:
        print("Running the search for best parameters of the selected models.")
        elapsed = measureTime(lambda : data_analysis.trainSearch())
        print("Finished best parameter search in {:.4f} seconds.".format(elapsed))

    if args.train_best:
        print("Training the selected models with best parameters.")
        elapsed = measureTime(lambda : data_analysis.trainBest())
        print("Finished training of the models with best parameters in {:.4f} seconds.".format(elapsed))

    if args.evaluate:
        print("Running the evaluation on test data.")
        elapsed = measureTime(lambda : data_analysis.evaluate(args.test_data))
        print("Finished test data evaluation in {:.4f} seconds.".format(elapsed))

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
