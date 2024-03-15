import os
from tqdm import tqdm

dataset_root = './raw_videos'
vid_file_lst = ['./splits/train_list.txt', './splits/val_list.txt', './splits/test_list.txt']
split_lst = ['training', 'validation', 'testing']
if not os.path.isdir(dataset_root):
    os.mkdir(dataset_root)
missing_vid_lst = []


for vid_file, split in zip(vid_file_lst, split_lst):
    if not os.path.isdir(os.path.join(dataset_root, split)):
        os.mkdir(os.path.join(dataset_root, split))
    with open(vid_file) as f:
        lines = f.readlines()
        for line in tqdm(lines):
            rcp_type, vid_name = line.replace('\n','').split('/')

            # download the video
            vid_url = 'www.youtube.com/watch?v='+vid_name
            vid_prefix = os.path.join(dataset_root, split, vid_name)
            os.system(' '.join(("yt-dlp -o", vid_prefix, vid_url)))

n = 0
for vid_file, split in zip(vid_file_lst, split_lst):
    with open(vid_file) as f:
        lines = f.readlines()
        for line in lines:
            rcp_type, vid_name = line.replace('\n','').split('/')
            vid_prefix = os.path.join('./raw_videos/', split, vid_name)
            new_prefix = vid_prefix + '.mp4'
            try: 
                os.rename(vid_prefix, new_prefix)
            except:
                missing_vid_lst.append('/'.join((split, line)))
                n += 1
                print(f'{n} missing videos')


with open('./missing_videos.txt', 'w') as f:
    for missing_vid in missing_vid_lst:
        f.write(missing_vid)


##clean
import os

with open('./splits/train_list.txt') as f:
    lines = f.readlines()
    for line in lines:
        rcp_type, vid_name = line.replace('\n','').split('/')
        vid_prefix = os.path.join('./raw_videos/training', vid_name)
        new_prefix = vid_prefix
        try: 
            os.rename(vid_prefix+'.mp4', new_prefix)
        except:
            print('error')