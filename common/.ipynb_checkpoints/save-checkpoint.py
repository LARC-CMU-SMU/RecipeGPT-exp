import os
import pickle
from datetime import datetime
dir_HugeFiles = '../data/'

def current_time():
    date_time = datetime.now()
    date = date_time.date()  # Gives the date
    time = date_time.time()  # Gives the time
    l_time = [date.month, date.day, '_',time.hour, time.minute, time.second]
    l_time = [str(ele) for ele in l_time]
    str_time = "".join(l_time)
    return str_time

def print_time():
    print(datetime.now())

def auto_save_pickle(obj, dir_path = dir_HugeFiles):
    dir_path = os.path.join(dir_path,'pickle')
    make_dir(dir_path)
    filename = str(datetime.now())+'.pickle'
    path_ = os.path.join(dir_path,filename)
    save_pickle(path_, obj)
    print('save to ' + path_)
    return path_

def save_pickle(filename, obj, overwrite = False):
    make_dir(filename)
    if os.path.isfile(filename) == True and overwrite == False:
        print('already exists'+filename)
    else:
        with open(filename, 'wb') as gfp:
            pickle.dump(obj, gfp, protocol=2)
            gfp.close()
        
def make_dir(filename):
    dir_path = os.path.dirname(filename)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print('make dir')
        
def load_pickle(filename):
    with open(filename, 'rb') as gfp:
        r = pickle.load(gfp)
    return r

def save_df(path, df_input, index_name):
    df = df_input.reset_index()
    df = df.rename(index=str, columns={"index": index_name})
    df.to_csv(path, index = False)
    
def auto_save_csv(df_input, path= 'csv/'):
    time = str(datetime.now())
    path_= path + time + '.csv'
    df_input_ = round(df_input,3)
    save_df(path_, df_input_, 'conditions')
    print('save to ' + path_)
    display(df_input_)
    
def save(filename, to_write, overwrite = False, print_= True):
    make_dir(filename)
    if os.path.isfile(filename) == True and overwrite == False:
        if print_:
            print('already exists'+filename)
    else:    
        with open(filename,'w') as f:
            f.write('%s' % to_write)
        if print_:
            print('saved '+filename)