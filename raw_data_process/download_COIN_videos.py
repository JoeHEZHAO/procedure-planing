import os, sys
import os.path as osp
import json
import tqdm
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))    # add parent dir
from paths import COIN_PATH

json_path = os.path.join(COIN_PATH, 'COIN.json')
coin_video_loc = os.path.join(COIN_PATH, 'videos')
if not os.path.exists(coin_video_loc):
    os.makedirs(coin_video_loc)
    
data = json.load(open(json_path, 'r'))['database']
youtube_ids = list(data.keys())
print(len(youtube_ids))
for youtube_id in tqdm.tqdm(youtube_ids):
    info = data[youtube_id]
    task_class = info['class']
    subset = info['subset']
    annotation = info['annotation']
    typer = info['recipe_type']
    url = info['video_url']
    st = info['start']
    ed = info['end']
    duration = info['duration']
    download_vid_name = '_'.join([task_class, str(typer), url.split('/')[-1]]) + '.mp4'
    download_substitue_name = '_'.join([task_class, str(typer), url.split('/')[-1]])

    os.system('youtube-dl -o ' + coin_video_loc + '/' + download_vid_name + ' -f best "' + url + '"')
    os.system('youtube-dl -o ' + coin_video_loc + '/' + download_substitue_name + ' --write-auto-sub --sub-lang en --convert-subs vtt --skip-download ' + url)
    raise