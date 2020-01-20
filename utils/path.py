import os
dir_HugeFiles = "/data/yueliu/RecipeAnalytics_201906/"
#print(os.getcwd())
if 'workspace' in os.getcwd():
    dir_HugeFiles = '../dir_HugeFiles/'

# save the cleaned data - basic clean
dir_json = dir_HugeFiles+ 'All_Recipe/json/'
dir_save = dir_HugeFiles+ 'All_Recipe/data.pickle'
