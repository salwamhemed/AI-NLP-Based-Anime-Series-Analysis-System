import glob
import pandas as pd



def load_subtitles(dataset_path):
 scripts= []
 episode_numbers=[]
 paths = glob(dataset_path+'/*.ass')
 for path in paths:
    with open(path,'r') as my_file:
      lines = my_file.readlines()
      lines = lines[25:]
      lines = [",".join(line.split(',')[9:]) for line in lines ]
     
    lines = [ line.replace('\n',' ') for line in lines ]
    lines = [ line.replace('\\N',' ') for line in lines ]
    script = " ".join(lines)
    episode_num= int(path.split('-')[-1].split('.')[0].strip())

    scripts.append(script)
    episode_numbers.append(episode_num)
  
 df = pd.DataFrame.from_dict({"episode": episode_numbers,"scripts": scripts})

 return df 
