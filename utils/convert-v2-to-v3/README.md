To convert a legacy v2 model to a v3 model, you can use the `convert_v2_to_v3.py` script. This script will take a v2 model and convert it to a v3 model. The script will also create a backup of the original v2 model in case you need to revert back to it.

## Usage
```
conda create -n convert-v2-to-v3
conda activate convert-v2-to-v3
pip install -r requirements.txt
python convert.py --savedmodel_dir ./models/v2_model --output_file ./models/v3_model.keras
```
