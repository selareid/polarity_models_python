# convert savedata of variation tests to csv
# baseline run as duplicate row in the output csv
# -1 in polarity marks 'FAILED' run

import numpy as np
from src.tasks import variation_task_helper as t_helper
from src.models.metric_functions import polarity_measure


# need this for data loading
def v_func_zero():
    pass


filename = "savedata filename"
is_goehring = False
# is_goehring = True


loaded_data = t_helper.load_runs(filename)
assert(loaded_data[0])

baseline = loaded_data[1]
results_by_variable = loaded_data[2]

# baseline is a tuple
# 0 - output
# 1 - input dictionary

# list[tuple[list]]
# | list - groups by parameter varied
# |--| tuple - length 2, output or input
#    |--| list - individual runs

# the sub-tuple runs are in matching order
# i.e. tuple[0][3] matches with tuple[1][3]

wanted_data = {}

for vary_group in results_by_variable:
    variable_data = []
    key_varied = vary_group[1][1]["key_varied"]

    for i in range(0, len(vary_group[0])):
        sol = vary_group[0][i]
        kvals = vary_group[1][i]

        if sol == "FAILURE":
            variable_data.append((kvals[key_varied], -1)) # -1 used as failure marker
            continue

        Nx = kvals["Nx"]

        if is_goehring:
            pm = polarity_measure(kvals["X"],
                                  sol.y[:Nx, -1],
                                  sol.y[Nx:, -1],
                                  Nx)
        else:
            pm = polarity_measure(kvals["X"],
                                  sol.y[:Nx, -1] + sol.y[Nx:2*Nx, -1] + sol.y[2*Nx:3*Nx, -1],
                                  sol.y[3*Nx:, -1],
                                  Nx)

        variable_data.append((kvals[key_varied], pm))

    wanted_data[key_varied] = variable_data

keys = wanted_data.keys()
data_out = []
header_arr = []

for key in keys:
    N = len(wanted_data[key])
    r1_keyvalue = [wanted_data[key][i][0] for i in range(N)]
    r2_polarity = [wanted_data[key][i][1] for i in range(N)]

    data_out.append(r1_keyvalue)
    data_out.append(r2_polarity)
    header_arr.append(key+"_value")
    header_arr.append(key+"_polaritymeasure")

np.savetxt(f"{filename}.csv", np.transpose(data_out), delimiter=',', fmt="%f", header=','.join(header_arr))
print(f"Saved to {filename}.csv")
