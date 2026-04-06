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

def create_gloss_subset(original_json: str, new_json: str, out_dir: str, out_name: str) -> None :
    """original_json file is one of the asl{100,300,1000,2000}.json files.
       new_json is the newly generated json for the asl citizen dataset.
    """
    with open(original_json) as f :
        original = json.load(f)
    with open(new_json) as f :
        new = json.load(f)
    output = []
    for original_entry in original :
        for new_entry in new :
            if original_entry["gloss"] == new_entry["gloss"].lower() :
                output.append(new_entry)
                break
                # print(output)
                # exit()
    with open(f"{out_dir}/{out_name}", 'w') as f :
        json.dump(output, f, indent=4)

if __name__ == "__main__" :
    root = "/home/pourquoi/repos/rtsl/src/backend/data_splits"
    create_gloss_subset(f"{root}/splits/asl100.json", f"{root}/asl_citizen/all_glosses.json", f"{root}/asl_citizen", "asl_citizens100.json")
    create_gloss_subset(f"{root}/splits/asl300.json", f"{root}/asl_citizen/all_glosses.json", f"{root}/asl_citizen", "asl_citizens300.json")
    create_gloss_subset(f"{root}/splits/asl1000.json", f"{root}/asl_citizen/all_glosses.json", f"{root}/asl_citizen", "asl_citizens1000.json")
    create_gloss_subset(f"{root}/splits/asl2000.json", f"{root}/asl_citizen/all_glosses.json", f"{root}/asl_citizen", "asl_citizens2000.json")
    # output = csv_to_json(["train.csv", "test.csv", "val.csv"])
    # with open("all_glosses.json", 'w') as g :
    #     json.dump(output, g, indent=4)

            # print(row["Video file"][:-4])
            # exit(0)
