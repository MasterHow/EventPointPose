# Evaluate Model
### Evaluate test dataset
```
cd ./evaluate
# change model name when init_point_model
python evaluate.py
```
You can output results on the test dataset with batch input.

The results will show in the terminal.

### Evaluate looping all h5 files
```
# change exp_name and init_point_model for each evaluation
python evaluate_all_h5.py
```
You can save an `.xlsx` file which containes each result for the `.h5` file in this folder.

### Evaluate on a single file
```
# change sub, session, mov for the specific movement, as well as exp_name and init_point_model
python evaluate_video_h5.py
```
You can save the `gif` and `videos` in this folder.
