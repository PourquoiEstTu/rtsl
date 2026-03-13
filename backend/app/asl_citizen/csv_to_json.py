import csv
import json
from sys import exit

def csv_to_json(files: list[str]) -> None :
    output = []
    gloss_to_idx = {}
    gloss_idx = 0
    for file in files :
        with open(file) as f :
            reader = csv.DictReader(f)
            # counter = 0
            for row in reader :
                # counter += 1
                # if counter >= 350 : break
                if row["Gloss"] not in gloss_to_idx :
                    gloss_to_idx[row["Gloss"]] = gloss_idx
                    gloss_idx += 1
                    # join removes numbers from gloss
                    entry = {"gloss": "".join(filter(lambda x: x.isalpha(), row["Gloss"])), 
                                "instances": [ {"video_id": row["Video file"][:-4], 
                                                "split": file[:-4]} ]}
                    output.append(entry)
                else :
                    entry = {"video_id": row["Video file"][:-4], "split": file[:-4]}
                    output[ gloss_to_idx[ row["Gloss"] ] ]["instances"].append(entry)
    return output

if __name__ == "__main__" :
    # output = []
    # gloss_to_idx = {}
    # with open("train.csv") as f :
    #     reader = csv.DictReader(f)
    #     gloss_idx = 0
    #     counter = 0
    #     for row in reader :
    #         counter += 1
    #         if counter >= 107 : break
    #         if row["Gloss"] not in gloss_to_idx :
    #             gloss_to_idx[row["Gloss"]] = gloss_idx
    #             gloss_idx += 1
    #             # join removes numbers from gloss
    #             entry = {"gloss": "".join(filter(lambda x: x.isalpha(), row["Gloss"])), 
    #                         "instances": [ {"video_id": row["Video file"][:-4], 
    #                                         "split": "train"} ]}
    #             output.append(entry)
    #         else :
    #             entry = {"video_id": row["Video file"][:-4], "split": "train"}
    #             output[ gloss_to_idx[ row["Gloss"] ] ]["instances"].append(entry)
    output = csv_to_json(["train.csv", "test.csv", "val.csv"])
    with open("all_glosses.json", 'w') as g :
        json.dump(output, g, indent=4)

            # print(row["Video file"][:-4])
            # exit(0)
