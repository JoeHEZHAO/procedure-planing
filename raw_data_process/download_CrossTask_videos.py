import os, sys
import os, sys
import os.path as osp
import csv
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))    # add parent dir
from paths import CROSSTASK_PATH

crosstask_primary_taskids = [23521, 59684, 71781, 113766, 105222, 94276, 53193, 105253, 44047, 76400, 16815, 95603, 109972, 44789, 40567, 77721, 87706, 91515]
crosstask_videourl_csv = osp.join(CROSSTASK_PATH, 'crosstask_release/videos.csv')
crosstask_video_loc = osp.join(CROSSTASK_PATH, 'crosstask_release/videos')
if not os.path.exists(crosstask_video_loc):
    os.makedirs(crosstask_video_loc)

with open(crosstask_videourl_csv, "r") as f:
    reader = csv.reader(f, delimiter="\t")
    for i, line in enumerate(reader):
        taskid, video_id, url = line[0].split(',')
        download_vid_name = '_'.join([taskid, video_id]) + '.mp4'
        download_substitue_name = '_'.join([taskid, video_id])

        if int(taskid) in crosstask_primary_taskids: 
            os.system('youtube-dl -o ' + crosstask_video_loc + '/' + download_vid_name + ' -f best "' + url + '"')
            os.system('youtube-dl -o ' + crosstask_video_loc + '/' + download_substitue_name + ' --write-auto-sub --sub-lang en --convert-subs vtt --skip-download ' + url)