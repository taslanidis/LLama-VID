import json
import re
import glob

if __name__ == "__main__":
    # create json file empty
    with open('/scratch-shared/scur0405/data/LLaMA-VID-Eval/ucf-crime/questions.json', 'r') as f:
        questions = json.load(f)

    with open('/scratch-shared/scur0405/data/LLaMA-VID-Eval/ucf-crime/test_questions.json', 'r') as f:
        test_questions = json.load(f)

    train_questions = []
    test_questions_ids = [q['question_id'] for q in test_questions]
    for q in questions:
        if q['question_id'] not in test_questions_ids:
            train_questions.append(q)

    output_list=[]
    for question in train_questions:
        insert={}
        insert["id"]=question['question_id']
        insert["video"]=f"/scratch-shared/scur0405/data/LLaMA-VID-Eval/ucf-crime/videos/{question['video_name']}"
        insert["conversations"]=[]

        first_inner_dict={}
        first_inner_dict["from"]="human"
        first_inner_dict["value"]=question['question']

        second_inner_dict={}
        second_inner_dict["from"]="gpt"
        second_inner_dict["value"]=question["answer"]
        
        insert["conversations"].append(first_inner_dict)
        insert["conversations"].append(second_inner_dict)
        output_list.append(insert)
 
    with open(f"/scratch-shared/scur0405/data/LLaMA-VID-Finetune/ucf-crime/crime_detection_qa_ft.json", "w+") as f:
        json.dump(output_list, f)