cd /scratch-shared/scur0405/data/LLaMA-VID-Eval
mkdir -p ucf-crime
cd ucf-crime

wget --no-check-certificate "https://www.dropbox.com/sh/75v5ehq4cdg5g5g/AABJtkTnNc8LcVTfH1gE_uFoa/Anomaly-Videos-Part-1.zip?dl=0" -O crime.zip
unzip crime.zip
rm crime.zip
wget --no-check-certificate "https://www.dropbox.com/sh/75v5ehq4cdg5g5g/AAAbdSEUox64ZLgVAntr2WgSa/Anomaly-Videos-Part-2.zip?dl=0" -O crime.zip
unzip crime.zip
rm crime.zip
wget --no-check-certificate "https://www.dropbox.com/sh/75v5ehq4cdg5g5g/AAAgpsRNSHI_BtRnSCxxR7j9a/Anomaly-Videos-Part-3.zip?dl=0" -O crime.zip
unzip crime.zip
rm crime.zip
wget --no-check-certificate "https://www.dropbox.com/sh/75v5ehq4cdg5g5g/AACeDPUxpB6sY2jKgLGzaEdra/Testing_Normal_Videos.zip?dl=0" -O normal.zip
unzip normal.zip
rm normal.zip

# get testing video ids
wget --no-check-certificate https://www.dropbox.com/sh/75v5ehq4cdg5g5g/AADjSOQ-NLIsCVNWT0Mrhp5ca/Temporal_Anomaly_Annotation_for_Testing_Videos.txt?dl=0 -O testing_anomaly_videos.txt

# create sample val
mkdir -p videos

# copy all parts in the same folder
mv Anomaly-Videos-Part-1/Abuse/*.mp4 videos/
mv Anomaly-Videos-Part-1/Arrest/*.mp4 videos/
mv Anomaly-Videos-Part-1/Arson/*.mp4 videos/
mv Anomaly-Videos-Part-1/Assault/*.mp4 videos/
mv Anomaly-Videos-Part-2/Burglary/*.mp4 videos/
mv Anomaly-Videos-Part-2/Explosion/*.mp4 videos/
mv Anomaly-Videos-Part-2/Fighting/*.mp4 videos/
mv Anomaly-Videos-Part-3/RoadAccidents/*.mp4 videos/
mv Anomaly-Videos-Part-3/Robbery/*.mp4 videos/
mv Anomaly-Videos-Part-3/Shooting/*.mp4 videos/
mv Testing_Normal_Videos_Anomaly/*.mp4 videos/


cd ~/LLaMA-VID/scripts
source process_ucf_questions.sh