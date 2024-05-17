import json
import re
import glob

if __name__ == "__main__":
    # create json file empty
    questions = []
    index = 1
    # for file in folder
    for filename in glob.glob("/scratch-shared/scur0405/data/LLaMA-VID-Eval/ucf-crime/videos/*.mp4"):
        filename = filename.split("/")[-1]
        mat = re.match(r'([^\d]*)(\d.*)', filename)
        answer = mat.groups()[0]
        answer = answer.replace("_", " ").strip()
        
        if answer == "RoadAccidents":
            answer = "Road Accident"
        elif answer == "Normal Videos":
            answer = "No crime commited in video"

        question = {
            'question_id': index,
            'video_name': filename,
            'question': 'What type of crime is commited in the video?',
            'answer': answer
        }
        questions.append(question)
        index += 1


    with open('/scratch-shared/scur0405/data/LLaMA-VID-Eval/ucf-crime/questions.json', 'w+') as f:
        json.dump(questions, f, indent=4)


    test_names = []
    with open('/scratch-shared/scur0405/data/LLaMA-VID-Eval/ucf-crime/testing_anomaly_videos.txt', 'r') as in_file:
        stripped = (line.strip() for line in in_file)
        for line in stripped:
            vname = line.split(" ")[0]
            test_names.append(vname)


    test_questions = []
    for question in questions:
        if question['video_name'] in test_names:
            test_questions.append(question)

    
    with open('/scratch-shared/scur0405/data/LLaMA-VID-Eval/ucf-crime/test_questions.json', 'w+') as f:
        json.dump(test_questions, f, indent=4)