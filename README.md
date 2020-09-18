## Overview
This is the github repo for 

```Helena H. Lee, Ke Shu, Palakorn Achananuparp, Philips Kokoh Prasetyo, Yue Liu, Ee-Peng Lim, and Lav R. Varshney. 2020. RecipeGPT: Generative Pre-training Based Cooking Recipe Generation and Evaluation System. In Companion Proceedings of the Web Conference 2020 (WWW 20 Companion), April 20-24, 2020, Taipei, Taiwan. ACM, New York, NY, USA, 4 pages. https://doi.org/10.1145/3366424.3383536 ```

Our paper can be downloaded at [arxiv](https://arxiv.org/abs/2003.02498)

Our poster can be downloaded at [here](https://drive.google.com/file/d/1DD5BJRRQZ4qATP_w0TjOfXYKN4EYs2zY/view?usp=sharing)

We also provide an online website that allows the users to generate cooking recipes at [https://recipegpt.org/](https://recipegpt.org/)

Please contact ```helenalee.bt01@gmail.com``` if you have any questions or problems.

We forked the [this](https://github.com/nshepperd/gpt-2) implemnetation in order to train and fine-tune gpt-2.


## Google colab
We create a notebook [google-colab-notebook](https://colab.research.google.com/drive/1_7KrN6UFsNId5uk23t8L4gMlCxUidkfj) to assist you re-produce our work.
We show examples of model training, model inference, and model evaluation. 
It's a simplified version of our notebook 6, 7, 9

## Project Structure
By default, the project assumes the following directory structure:

 
    +-- data                                    # Files that are within GitHub's file size limit
    ¦   +-- vocab.bin                           # A word embedding model, will be used in utils.tree
    ¦   
    ¦   +-- recipe1M_example                    # Examples of testing data, n=5
    ¦       +-- test/X                          # Inputs in testing set (directory of .txt files)
    ¦       +-- test/y                          # Human-written outputs in testing set (directory of .txt files)
    ¦   
    ¦   
    +-- big_data                                # Files that exceeds GitHub's file size limit
    ¦   +-- layer1.pickle                       # Recipe1M from http://pic2recipe.csail.mit.edu/  (Please click Layers (381 MiB))
    ¦   +-- recipe1M_ny.pickle                  # Recipe1M processed with ny-times-parser        
    ¦   +-- data.pickle                         # The data after our own data pre-processing
    ¦ 
    ¦   +-- food_taxonomy.txt                   # Ingredients database
    ¦   +-- database.pickle                     # Ingredients root noun database
    ¦   
    ¦   
    +-- recipe1M_1218                           # files created by notebook 3 using data.pickle
    ¦   +-- chunk.train                         # Essential training data
    ¦   +-- chunk.val                           # Essential validation data
    ¦   +-- test/X                              # Inputs in testing set (directory of .txt files)
    ¦   +-- test/y                              # Human-written outputs in testing set (directory of .txt files)
    ¦   
    ¦   
    +-- analysis                                
    ¦   +-- notebook 1-1, 1-2, 2, 3             # Useful for data pre-processing
    ¦   +-- notebook 4, 5                       # Useful for analyzing the generated texts
    ¦   +-- notebook 9                          # Compare the generated texts with human-written texts
    ¦   +-- notebook 10                         # Explain how we convert the users' inputs to model input
    ¦   +-- notebook 11                         # Explain the 'compare' feature on the website
    ¦   
    ¦ 
    +-- training                                
    ¦   +-- gpt-2                               # The source code modified from OpenAI GPT-2
    ¦       +--src/load_dataset_pad.py              # Padding and fields shuffing
    ¦       +--src/conditional_gen_web.py           # Input .txt files and receive the output in .txt files
    ¦       +--train_ppl_pickle.py                  # The main script for fine-tuning with recipe data
    ¦       +--train_ppl_scratch.py                 # The main script for training from scratch with recipe data
    ¦       ...
    ¦       
    ¦                                           # Details of our experiments as described in the paper
    ¦   +-- notebook 6                          # Commands of fine-tuning/training the model
    ¦   +-- notebook 7                          # Ask the model to generate the title/ingredients/instructions 
    ¦   +-- notebook 8                          # Evaluate the model perplexity
    ¦ 
    +-- common                                  # Import numpy, pickle, ... etc common packages
    +-- utils                                   # Some modules related to model evaluation



## Download related files
We share the dataset, model, and related files at [google-drive](https://drive.google.com/drive/folders/1h82H1QEnBHCetSYT-yQr_usBEo7E6cIX?usp=sharing)

Please download and put them under the correct directory.
```
RecipeGPT-exp/big_data/
RecipeGPT-exp/training/gpt-2/models/
RecipeGPT-exp/recipe1M_1218/

```
* We utilize ```food_taxonomy.txt``` from [Here](https://www.researchgate.net/publication/288838055_Simple_food_taxonomy_compiled_from_Wikipedia_pages) to create an ingredient database . (Download your own or from our google-drive)
* We utilize ```vocab.bin```, the word embedding trained by [Salvador](http://pic2recipe.csail.mit.edu/im2recipe.pdf) (Already in github)
* We utilize  ```layer1.json``` , which contains the textual content of [Recipe1M](http://pic2recipe.csail.mit.edu/)(Please click Layers (381 MiB) and download your own)

You may need [these files](https://drive.google.com/drive/u/2/folders/1laFJaKjow8Jq8tPATwHpzVl8i3FKN39s) if you want to reproduce notebook 9.

## Environment
```
conda create -n recipegpt python=3.5 anaconda
pip install -r requirements.txt
```
tensorflow version:
```
pip install tensorflow-gpu==1.12.0
pip install tensorflow==1.12.0
or use the default version in colab
```
## 
