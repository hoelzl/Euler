# %%
import json
from pathlib import Path
from pprint import pprint

from nbex.interactive import session
import numpy as np
from mip import *

# %%
def read_json(path):
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    numbers = np.array([[int(c) for c in s] for s, _ in data])
    num_correct = np.array([n for _, n in data])
    return numbers, num_correct


# %%
root_path = Path("data/euler_185")

# %%
digits, num_correct = read_json(root_path / "data_16.json")
digits.shape, num_correct.shape

# %%
m = Model(sense=MAXIMIZE)

# %%
vars = m.add_var_tensor((10, digits.shape[1]), name="d", var_type=BINARY)
vars.shape

# %%
## Constraints for each column:
for col in range(vars.shape[1]):
    m += xsum(vars[:, col]) == 1

# %%
## Constraints for each row:
for row in range(digits.shape[0]):
    m += (
        xsum(vars[digits[row, col], col] for col in range(vars.shape[1]))
        == num_correct[row]
    )

# %%
if session.is_interactive:
	for line in [str(c.expr) for c in m.constrs]:
		print(line)

# %%
m.optimize()

# %%
def get_values():
    return np.array(
        [vars[i, j].x for i in range(vars.shape[0]) for j in range(vars.shape[1])]
    ).reshape(10, -1)

# %%
def get_result():
	return "".join(str(val) for val in np.argmax(get_values(), axis=0))

# %%
get_result()

# %%
def is_row_valid(row):
	row_sum = np.sum([vars[digits[row, col], col].x for col in range(vars.shape[1])])
	return row_sum == num_correct[row]

# %%
is_row_valid(0), is_row_valid(1)

# %%
def are_all_rows_valid():
	return all(is_row_valid(row) for row in range(digits.shape[0]))

# %%
are_all_rows_valid()

# %%
if __name__ == "__main__":
	if are_all_rows_valid():
		print(f"A possible result is {get_result()}.")
	else:
		print("No solution could be found.")
	
# %%

