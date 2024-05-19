import json
import re
import glob

if __name__ == "__main__":
    # create json file empty
    with open('./data/LLaMA-VID-Eval/ucf-crime/questions.json', 'r') as f:
        questions = json.load(f)

    with open('./data/LLaMA-VID-Eval/ucf-crime/test_questions.json', 'r') as f:
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
        insert["video"]=f"ucf-crime/videos/{question['video_name']}"
        insert["conversations"]=[]

        first_inner_dict={}
        first_inner_dict["from"]="human"
        first_inner_dict["value"]="<image>\n" + question['question']

        second_inner_dict={}
        second_inner_dict["from"]="gpt"
        second_inner_dict["value"]=question["answer"]
        
        insert["conversations"].append(first_inner_dict)
        insert["conversations"].append(second_inner_dict)
        output_list.append(insert)
 
    print("Big dataset size:", len(output_list))
    with open(f"./data/LLaMA-VID-Finetune/ucf-crime/crime_detection_qa_ft.json", "w+") as f:
        json.dump(output_list, f)

    small_videos_list = []
    from decord import VideoReader, cpu

    for question in output_list:
        video_path: str = "./data/LLaMA-VID-Eval/" + question['video']
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        fps = round(vr.get_avg_fps())
        print(f"Current Video Frames: {total_frame_num} and FPS: {fps} ---- {video_path.split('/')[-1]}")
        frame_idx = [i for i in range(0, len(vr), fps)]
        if len(frame_idx) <= 120:
            small_videos_list.append(question)


    print("Small dataset size:", len(small_videos_list))
    with open(f"./data/LLaMA-VID-Finetune/ucf-crime/crime_detection_small_qa_ft.json", "w+") as f:
        json.dump(small_videos_list, f)