import numpy as np


def calc_wighted_average(im1, imp1_multiplier, im2, imp2_multiplier):
    return round((((im1 * imp1_multiplier) + (im2 * imp2_multiplier)) / (imp1_multiplier+imp2_multiplier)), 3)


def calc_impurity(data):
    if len(np.unique(data[:, 0])) > 2:
        sorted_data = data[data[:, 0].argsort()]
        main_dict = {}
        for i in range(1, len(sorted_data)):
            first_number = sorted_data[i-1, 0]
            second_number = sorted_data[i, 0]
            avg = (first_number + second_number) / 2
            true_xs = data[data[:, 0] < avg]
            count_true_xs = len(true_xs)
            true_xs_true_ys = len(true_xs[true_xs[:, 1] == True])
            true_xs_false_ys = len(true_xs[true_xs[:, 1] == False])
            imp1 = round(1 - ((true_xs_true_ys / count_true_xs) **
                         2) - ((true_xs_false_ys / count_true_xs) ** 2), 3)
            false_xs = data[data[:, 0] > avg]
            count_false_xs = len(false_xs)
            false_xs_true_ys = len(false_xs[false_xs[:, 1] == True])
            false_xs_false_ys = len(false_xs[false_xs[:, 1] == False])
            imp2 = round(1 - ((false_xs_true_ys / count_false_xs) **
                         2) - ((false_xs_false_ys / count_false_xs) ** 2), 3)
            main_dict[str(avg)] = (calc_wighted_average(
                imp1, count_true_xs, imp2, count_false_xs))
        return {min(main_dict, key=main_dict.get): main_dict[min(main_dict, key=main_dict.get)]}
    else:
        true_xs = data[data[:, 0] == True]
        count_true_xs = len(true_xs)
        if count_true_xs == 0:
            imp1 = 0
        else:
            true_xs_true_ys = len(true_xs[true_xs[:, 1] == True])
            true_xs_false_ys = len(true_xs[true_xs[:, 1] == False])
            imp1 = round(1 - ((true_xs_true_ys / count_true_xs) **
                         2) - ((true_xs_false_ys / count_true_xs) ** 2), 3)
        false_xs = data[data[:, 0] == False]
        count_false_xs = len(false_xs)
        if count_false_xs == 0:
            imp2 = 0
        else:
            false_xs_true_ys = len(false_xs[false_xs[:, 1] == True])
            false_xs_false_ys = len(false_xs[false_xs[:, 1] == False])
            imp2 = round(1 - ((false_xs_true_ys / count_false_xs) **
                         2) - ((false_xs_false_ys / count_false_xs) ** 2), 3)
        return calc_wighted_average(imp1, count_true_xs, imp2, count_false_xs)


df = np.array([[1, 0, 18, 1],
               [1, 1, 15, 1],
               [0, 1, 65, 0],
               [0, 0, 33, 0],
               [1, 0, 37, 1],
               [0, 1, 45, 1],
               [0, 1, 50, 0],
               [1, 0, 75, 0],
               [1, 0, 67, 1],
               [1, 1, 60, 1],
               [0, 1, 55, 1],
               [0, 0, 69, 0],
               [0, 0, 80, 0],
               [0, 1, 87, 1],
               [1, 0, 38, 1]
               ])

print(calc_impurity(df[:, [0, -1]]))
print(calc_impurity(df[:, [1, -1]]))
print(calc_impurity(df[:, [2, -1]]))

true_side_df = df[df[:, 2] < 68]
print(calc_impurity(true_side_df[:, [0, -1]]))
print(calc_impurity(true_side_df[:, [1, -1]]))

false_side_df = df[df[:, 2] > 68]
print(calc_impurity(false_side_df[:, [0, -1]]))
print(calc_impurity(false_side_df[:, [1, -1]]))
