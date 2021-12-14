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
    
    print("You want to use the california.joblib")

    # Create the model for boston
    if (not os.path.isfile(filename)):
        
        print("The .joblib don't exist so we will create the model and save it.")

        california = datasets.fetch_california_housing()

        df = pd.DataFrame(california.data, columns = california.feature_names)

        X_train, X_test, Y_train, Y_test = train_test_split(df,
                california.target, test_size = 0.001, random_state=5)

        Model = LinearRegression()
        Model.fit(X_train, Y_train)

        y_train_predict = Model.predict(X_train)
        rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
        r2 = r2_score(Y_train, y_train_predict)
        
        print("\n\n\nWe have the following Result for the model :")
        print("Train RMSE : ", rmse, " and R2 : ", r2, ".")

        y_test_predict = Model.predict(X_test)
        rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
        r2 = r2_score(Y_test, y_test_predict)
        print("Test RMSE : ", rmse, " and R2 : ", r2, ".")


        print("\n\n\nWe can nox save the model.")
        joblib.dump(Model, filename)
        
        print("\n\nThe result for the test [3.25, 28.0, 5.5, 1.11, 1162.0, 2.08, 34.00, -118.35] is : ", Model.predict([[53.25, 28.0, 5.5, 1.11, 1162.0, 2.08, 34.00, -118.35]]))

    else:

        print("The .joblib exist so we just will load and use it.")
        
        Model = joblib.load(filename)


        california = datasets.fetch_california_housing()

        df = pd.DataFrame(california.data, columns = california.feature_names)

        X_train, X_test, Y_train, Y_test = train_test_split(df,
                california.target, test_size = 0.001, random_state=5)

        print("\n\nThe result for the test [3.25, 28.0, 5.5, 1.11, 1162.0, 2.08, 34.00, -118.35] is : ", Model.predict([[3.25, 28.0, 5.5, 1.11, 1162.0, 2.08, 34.00, -118.35]]))
else:
        Model = joblib.load(filename)

print("\n\n\nSo we have load the model : ", filename)
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
        "\tfloat feature["  + str(len(Model.coef_)) + "] = {3.25, 28.0, 5.5, 1.11, 1162.0, 2.08, 34.00, -118.35};\n\n" + \
        "\tprintf(\"The result is : \%f\\n\", prediction(feature, 8));\n\n}\n"

out_filename = "prediction.c"

with open(out_filename, "w") as f:
        f.write(code_c)
