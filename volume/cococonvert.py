import pandas as pd 
import json
import os 


BOX_COLS = ["x", "y", "z", "w", "l", "h", "rx", "ry", "rz"]


def convert_csv_labels(labels: str, file: str):
    df = pd.read_csv(labels)

    df['bbox3d'] = df[BOX_COLS].values.tolist()
    df.drop(BOX_COLS, axis=1, inplace=True)
    df.reset_index(inplace=True)
    df.rename({'index':'id','class_':'category_id', 'imgnr':'image_id'}, axis=1, inplace=True)

    body = {
        "images": [{'id':id_} for id_ in df.image_id.unique().tolist()],
        "annotations":df.to_dict(orient="records"),
        "categories":[
            {'id': 0, 'name':'haddock'},
            {'id': 1, 'name':'hake'},
            {'id': 2, 'name':'herring'},
            {'id': 3, 'name':'mackerel'},
            {'id': 4, 'name':'redgurnard'},
            {'id': 5, 'name':'whiting'},
        ]
    }

    with open(file, "w") as f:
        json.dump(body, f, indent=4)


def convert_csv_outputs(outputs: str, file: str):
    df = pd.read_csv(outputs)
    df['bbox3d'] = df[BOX_COLS].values.tolist()
    df.drop(BOX_COLS, axis=1, inplace=True)
    df.rename({'class_':'category_id', 'imgnr':'image_id', 'conf':'score'}, axis=1, inplace=True)
    df.to_json(file, orient="records")


if __name__ == '__main__':
    mapdir = '/mnt/deepvol/mapdir'
    import glob 
    convert_csv_labels(os.path.join(mapdir, "nogit_val_labels.csv"), "jsons/nogit_val_labels.json")

    for dir_ in glob.glob(os.path.join(mapdir, "nogit_output_*.csv")):
        print(dir_)
        convert_csv_outputs(dir_, "jsons/"+os.path.basename(dir_).replace('.csv','.json'))