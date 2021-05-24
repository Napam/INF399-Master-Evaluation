import pandas as pd 
import json 

BOX_COLS = ["x", "y", "z", "w", "l", "h", "rx", "ry", "rz"]

def convert_csv_labels(labels: str):
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

    with open("nogit_coco_labels.json", "w") as f:
        json.dump(body, f, indent=4)


def convert_csv_outputs(outputs: str):
    df = pd.read_csv(outputs)
    
    df['bbox3d'] = df[BOX_COLS].values.tolist()
    df.drop(BOX_COLS, axis=1, inplace=True)
    df.rename({'class_':'category_id', 'imgnr':'image_id', 'conf':'score'}, axis=1, inplace=True)
    df.to_json("nogit_coco_outputs.json", orient="records")

    

if __name__ == '__main__':
    convert_csv_labels('nogit_train_labels.csv')
    convert_csv_outputs('nogit_train_output.csv')