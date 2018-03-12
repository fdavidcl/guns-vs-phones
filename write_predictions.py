import glob, os

import pandas as pd
import numpy as np

def write_predictions(predictions, filename = "output.csv"):
    ids = [os.path.basename(f) for f in glob.glob("Test/*")]
    solution = pd.DataFrame(data = {"ID": ids, "Ground_Truth": np.argmax(predictions, axis = 1)})
    solution.to_csv(filename, index = False, columns = ["ID", "Ground_Truth"])

    return filename
