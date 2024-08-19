#!/bin/bash

source ~/.virtualenvs/models/bin/activate

notebooks=("feature_user_size_lstm" "feature_tlds_lstm")

for notebook in "${notebooks[@]}"
do
    input_notebook="${notebook}.ipynb"
    output_notebook="${notebook}_output.ipynb"

    # Execute the notebook using nbconvert with the specified kernel
    jupyter nbconvert --to notebook --execute "$input_notebook" --output "$output_notebook" --ExecutePreprocessor.kernel_name=models

    # Check if the execution was successful
    if [ $? -eq 0 ]; then
        echo "Successfully executed $input_notebook"
    else
        echo "Error executing $input_notebook"
        exit 1  # Exit the script if there is an error
    fi
done

echo "All notebooks have been executed successfully."
deactivate