## Overview
This is the github repo for Helena H. Lee, Ke Shu, Palakorn Achananuparp, Philips Kokoh Prasetyo, Yue Liu, Ee-Peng Lim, and Lav R. Varshney. 2020. RecipeGPT: Generative Pre-training Based Cooking Recipe Generation and Evaluation System. In Companion Proceedings of the Web Conference 2020 (WWW 20 Companion), April 20-24, 2020, Taipei, Taiwan. ACM, New York, NY, USA, 4 pages. https://doi.org/10.1145/3366424.3383536 

This paper can be downlaoded at [arxiv](https://arxiv.org/pdf/1909.07881.pdf)

Please contact ```helenalee.bt01@gmail.com``` if you have any questions or problems.

## Slides 
Our poster presented in WWW'20 could be found at [here](https://drive.google.com/file/d/1DD5BJRRQZ4qATP_w0TjOfXYKN4EYs2zY/view?usp=sharing)

## Project Structure
By default, the project assumes the following directory structure:

 
    +-- data                                    # Files that we save
    ¦   +-- vocab.bin                          # The textual description of recipe, AMT annotation, and nutritional properties
    ¦   +-- Recipe54k-trained embeddings        # Some pickle files
    ¦   +-- combined.csv                        # 1000 recipes with crowdsourcing annotations
    ¦   +-- ... 
    ¦ 
    +-- RQ1                                     # How we conduct data preprocessing and crowdsourcing to answer the Research Question 1
    ¦   +-- ... 
    ¦ 
    +-- RQ2                                     # Models trained to answer the Research Question 2 and saved to csv/ and pickle/
    ¦   +-- RQ2_original                        # How we prepare the results on paper, it is not well-structured and requires dic_20190819.pickle
    ¦   ¦   +-- ...
    ¦   +-- RQ2_reproducible                    # We selectively reproduce the best models in our study and re-organize the notebooks. It requires dic_20191203.pickle
    ¦   ¦   +-- Best models in RQ2.ipynb        # Best non-nutritional model: NB-BoW + LR
    ¦   ¦                                       # Best overall model:         Pre-trained GloVe + Nutritional information + LGBM
    ¦   ¦                                       # Second best overall model:  NB-BoW + Nutritional information + LR
    ¦   ¦                                       # Nutrition-only model:       Nutritional information + LR
    +-- pickle     
    +-- csv     
    +-- ...

## Dataset
We download use the textual content of [Recipe1M](https://arxiv.org/pdf/1909.07881.pdf)
The file is called ```layer1.json```


## Environment
```
conda create -n recipegpt python=3.5 anaconda
pip install tensorflow-gpu==1.12.0 or pip install tensorflow==1.12.0
pip install -r requirements.txt
```
## 
