# File credit: Vincent Tjeng. Modified by Maksym Andriushchenko.
import sys
sys.path.append('..')  # to import data.py from the parent directory
import data
import pandas as pd


def preprocess_summary_file(dt, pd_y_test, get_last=True):
    dt.SampleNumber -= 1
    dt.PredictedIndex -= 1  # To compensate for Julia's numeration that starts from 1.
    dt = dt.drop_duplicates(
        subset="SampleNumber", keep="last" if get_last else "first"
    ).set_index("SampleNumber").sort_index().join(pd_y_test)
    dt["IsCorrectlyPredicted"] = dt.PredictedIndex == dt.TrueIndex
    dt["ProcessedSolveStatus"] = dt["SolveStatus"].apply(process_solve_status)

    dt["BuildTime"] = dt["TotalTime"] - dt["SolveTime"]
    return dt


def get_dt(filename, dataset):
    dt = pd.read_csv(filename)
    _, _, _, y_test = data.get_dataset(dataset)
    pd_y_test = pd.DataFrame({'TrueIndex': y_test.argmax(1)})
    return preprocess_summary_file(dt, pd_y_test)


def process_solve_status(s):
    # Note: Unbounded is impossible, since with eps=0 we recover the original point.
    # So it boils down to just infeasible or infeasible, which is what we want.
    if s == "InfeasibleOrUnbounded" or s == "Infeasible":
        return "ProvablyRobust"
    # `UserLimit` is due to the timeout. Error never happened to me yet.
    elif s == "UserLimit" or s == "Error":
        return "StatusUnknown"
    # `UserObjLimit` is due to BestObjStop=eps. `Optimal` often just means a misclassified point.
    # Note: ObjectiveBound == 0 currently, since this fixes the bug with the L2 norm.
    elif s == "UserObjLimit" or s == "Optimal":
        return "Vulnerable"  # all misclassified points are also marked as "Vulnerable"
    else:
        raise ValueError("Unknown solve status {}".format(s))


def summarize_processed_solve_status(dt):
    dt_summary = dt.groupby("ProcessedSolveStatus").TotalTime.count() / len(dt)

    if "Vulnerable" not in dt_summary.index:
        dt_summary["Vulnerable"] = 0
    dt_summary["Robust Error, LB"] = dt_summary["Vulnerable"]
    if "StatusUnknown" not in dt_summary.index:
        dt_summary["StatusUnknown"] = 0
    dt_summary["Robust Error, UB"] = dt_summary["Robust Error, LB"] + dt_summary["StatusUnknown"]

    if "UserLimit" not in dt_summary.index:
        dt_summary["UserLimit"] = 0
    dt_summary["TimeoutNumber"] = dt[dt['SolveStatus'] == 'UserLimit'].count()['TargetIndexes'] / len(dt)

    dt_natural_summary = dt.groupby("IsCorrectlyPredicted").TotalTime.count() / len(dt)
    try:
        dt_summary["RegularError"] = dt_natural_summary[False]
    except:
        dt_summary["RegularError"] = 0

    dt_summary["TotalTime"] = dt['TotalTime'].sum()
    dt_summary["Count"] = dt.count()["TargetIndexes"]

    return dt_summary


def get_summary(filename, dataset):
    dt = get_dt(filename, dataset)
    dt_summary = summarize_processed_solve_status(dt)
    return dt_summary


def get_bounds_pointwise(filename, dataset):
    dt = get_dt(filename, dataset)
    is_non_robust = dt["ProcessedSolveStatus"] == "Vulnerable"
    is_provably_robust = (dt["ProcessedSolveStatus"] != "Vulnerable") * (dt["ProcessedSolveStatus"] != "StatusUnknown")
    return is_non_robust.values, is_provably_robust.values
