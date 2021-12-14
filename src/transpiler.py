# Import

import os
import sys

import numpy as np
import joblib
import pandas as pd

import sklearn
from sklearn import datasets
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Setup

filename = "california.joblib"
Model = None
next_is_filename = False
for elem in sys.argv:

    if (elem == "--filename"):
        next_is_filename = True
    elif (next_is_filename == True):
        filename = elem

if (filename[-7:] != ".joblib"):
    print("Erreur in the filename you have to use a .joblib file.")

if (filename == "california.joblib"):

    # Create the model for boston
    if (not os.path.isfile(filename)):

        california = datasets.fetch_california_housing()

        df = pd.DataFrame(california.data, columns = california.feature_names)

        X_train, X_test, Y_train, Y_test = train_test_split(df,
                california.target, test_size = 0.001, random_state=5)

        Model = LinearRegression()
        Model.fit(X_train, Y_train)

        y_train_predict = Model.predict(X_train)
        rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
        r2 = r2_score(Y_train, y_train_predict)
        print("Train RMSE : ", rmse, " and R2 : ", r2, ".")

        y_test_predict = Model.predict(X_test)
        rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
        r2 = r2_score(Y_test, y_test_predict)
        print("Test RMSE : ", rmse, " and R2 : ", r2, ".")


        joblib.dump(Model, filename)

    else:

        Model = joblib.load(filename)


        california = datasets.fetch_california_housing()

        df = pd.DataFrame(california.data, columns = california.feature_names)

        X_train, X_test, Y_train, Y_test = train_test_split(df,
                california.target, test_size = 0.001, random_state=5)

        print(Model.predict(X_test))
        print("The result : ", Model.predict([[0.1, 0.1, 0.1, 0.1,0.1, 0.1, 0.1, 0.1]]))
else:
        Model = joblib.load(filename)

print("So we have load the model : ", filename)
print("\nAnd the coefficients of the model are : ", Model.coef_)
print("And the intercept is : ", Model.intercept_)

code_c = "#include <stdio.h>\n\n" + \
        "float prediction(float *features, int n_feature)\n" + \
        "{\n" +  \
        "\tfloat coef[" + str(len(Model.coef_)) + "] = {"

for i in range(len(Model.coef_)):

    if ((i != 0) and (i % 4 == 0)):
        code_c += "\n\t\t"

    code_c += str(round(Model.coef_[i], 5)) + ","
    
    
code_c = code_c[:-1] + "};\n"

code_c += "\tfloat result = 0.0;\n\t" + \
        "for(int i = 0; i < n_feature; i++)\n\t" + \
        "{\n\t\t" + \
        "result += features[i] * coef[i];\n\t\t" + \
        "\n\t}\n\treturn result + " + str(round(Model.intercept_, 5)) + ";\n}\n\n" + \
        "int main(void)\n" + \
        "{\n" + \
        "\tfloat feature["  + str(len(Model.coef_)) + "] = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};\n\n" + \
        "\tprintf(\"The result is : \%f\\n\", prediction(feature, 8));\n\n}\n"

out_filename = "prediction.c"

with open(out_filename, "w") as f:
        f.write(code_c)
