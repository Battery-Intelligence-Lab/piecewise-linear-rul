"""run a piecewise linear trial"""
from pandas import read_csv
from numpy import round

from soh_prognosis_plr import fixdata, generate_cell_numbers, piecewise_linear_model_wrapper
from results import calculate_results, disp_profile_results


if __name__ == "__main__":

    repeats = 10
    ntraincells = 50  # will train on a given number then test on the rest

    # Load data (you'll have to sort the data yourself yourself)
    feature_data = fixdata(read_csv("feature_data.csv"))

    # form trialsample numbers
    cellnumbers = generate_cell_numbers(number_of_cells=len(feature_data), repeats=repeats)

    predictions_plr = []
    for rep, cell_numbers in enumerate(cellnumbers):

        predictions_plr = predictions_plr + piecewise_linear_model_wrapper(
            input_feature_data=feature_data,
            cell_numbers=cellnumbers[rep],
        )

        percentage = round((rep + 1) * 100 / repeats)
        print(f" ... {percentage}% complete")

    results_plr = calculate_results(results_list=predictions_plr)
    disp_profile_results("Piecewise linear regression", results_plr)
