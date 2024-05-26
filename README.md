# Seeking the limitations of "LLaMA-VID: An Image is Worth 2 Tokens in Large Language Models"
### A. Tragadouras, E.G. Lionis, T. Aslanidis, V. Karlis, O. Neut
---

In this repository, we have code to reproduce and extend the findings of the paper titled ["LLaMA-VID: An Image is Worth 2 Tokens in Large Language Models"](https://arxiv.org/abs/2311.17043).

## Structure of Repository
The structure of the repository has tried to be as modularized as possible and stay as original structure of the original authors to not cause any problems with dependencies. The repository must follow the following structure:
```
repo
|-- readme.md  
|
|-- blogpost.md
|
|-- src
|   |-- llamavid
|   |-- scripts
|   |-- work_dirs
|   │   |-- llama-vid
|   │   │   |-- llama-vid-13b-full-336
|   │   │   |-- ...
|   |   |-- llama-vid-7b-full-224-video-fps-1_grounding_finetuning
|   |   |-- llama-vid-7b-full-224-video-fps-1-small-ucf-finetuning
|   |   |-- ... (more folders with the output)
|   |-- model_zoo
|   |   |-- LLM
|   |   │   |-- vicuna
|   |   │   │   |-- 7B-V1.5
|   |   │   │   |-- 13B-V1.5
|   |   |-- LAVIS
|   |   │   |-- eva_vit_g.pth
|   |   │   |-- instruct_blip_vicuna7b_trimmed.pth
|   |   │   |-- instruct_blip_vicuna13b_trimmed.pth
|   |-- new_datasets_processing
|   |   |-- animal_dataset
|   |   |-- ucf_dataset
|   |-- data  
|   |   |   |-- LLaMA-VID-Eval
|   |   |   |   |-- animal-kingdom
|   |   |   |   |-- ucf-crime
|   |   |   |   |-- gqa
|   |   |   |   |-- ... (more folders)
|   |   |   |-- LLaMA-VID-Pretrain
|   |   |   |   |-- animal-kingdom
|   |   |   |   |-- ucf-crime  
|   |   |   |   |-- gqa
|   |   |   |   |-- ... (more folders)
|   |   |   |-- LLaMA-VID-Finetune
|   |   |   |   |-- animal-kingdom
|   |   |   |   |-- ucf-crime  
|   |   |   |   |-- gqa
|   |   |   |   |-- ... (more folders)
|
|-- demos
|   |-- test_1.ipynb
|   |-- experiment_2.ipynb
|-- pyproject.toml
|-- requirements_llama_three.txt
|-- install_llama_environment.job
|-- install_environment.job
```


More details for some files/folders are the following:
- **readme.md**:      Description of the repository
- **blogpost.md**:    Blogpost report
- **src**:            contains the main project files
    - **model_zoo**:    Folder with all imported weights for the LLM and vision encoder
    - **llamavid**:     The code that has the model architecture and will be executed by the scripts
    - **scripts**:      The executable code
    - **work_dirs**:    The output of training, fine-tuning and evaluation. If a pre-train model is used, then this folder has to have the weights.
    - **llama-vid**:    The weights of the llama-vid model from the original authors
    - **llama-vid-7b-full-224-video-fps-1_grounding_finetuning**:
                        The weights for the finetuning with the animal grounding dataset
    - **llama-vid-7b-full-224-video-fps-1-small-ucf-finetuning**:
                        The weights for the finetuning with the small-ucf dataset
    - **new_datasets_processing**:
                        The code for processing the new dataset
    - **data**:         The folder with all data
        - **LLaMA-VID-Eval**:       The data for evaluation purposes
        - **LLaMA-VID-Pretrain**:   The data for pre-train purposes
        - **LLaMA-VID-Finetune**:   The data for finetuning
- **requirements_llama_three.txt**:
- **install_llama_environment.job**: Executable to install the conda environment in a server
- **install_environment.job**: Executable to install the conda environment in a server


## Download Data
The data that has been used in this project, can be split into reproducible data and the extension data. 

### Original benchmarks
The reproducible data is the benchmarks that the authors used and can be found in the [original repository](https://github.com/dvlab-research/LLaMA-VID?tab=readme-ov-file#dataset). In order to download this dataset, one has to follow the steps that the original authors describe for the images and short video evaluations.

### Extended data
The extended data are data from two different datasets. The one dataset is [animal kingdom dataset](https://sutdcv.github.io/Animal-Kingdom/) which we took the Video Grounding dataset, where we used it for fine-tuning and inference purposes. This data contains 50 hours of annotated videos to localize relevant animal behavior segments in long videos. In order to download them you need to request access through the [form](https://forms.office.com/pages/responsepage.aspx?id=drd2NJDpck-5UGJImDFiPVRYpnTEMixKqPJ1FxwK6VZUQkNTSkRISTNORUI2TDBWMUpZTlQ5WUlaSyQlQCN0PWcu) that they have in their page and get the sharepoint link. For a similar reason, the second dataset is [UCF-Crime dataset](https://paperswithcode.com/dataset/ucf-crime) and it consists of footage from 1900 long and untrimmed real-world surveillance videos, with 13 activities. In order to download this benchmark you can execute `scripts/download_ucf_crime.sh` script or manually download it from the Dropbox URL that they provide on their page.

## Download Pre-trained Weights
The steps to download the pre-trained weights are split into two 2 parts. The weights are needed for the performing both training and inference on the model. There also the weights have passed through finetuning and are not necessary if you are planning to retrain from the start of the model.

### Necessary pre-trained weights
There is a [section](https://github.com/dvlab-research/LLaMA-VID?tab=readme-ov-file#pretrained-weights) from the original authors to describe how to download the necessary weights. You must put them under model_zoo as defined in the [structure](https://github.com/taslanidis/LLama-VID?tab=readme-ov-file#structure-of-repository). We used the [vicuna-7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5) data out of the vicuna and have to be placed under `src/model_zoo/LLM/vicuna`. Also we downloaded the [EVA-ViT-G](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth), [QFormer-7b](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/InstructBLIP/instruct_blip_vicuna7b_trimmed.pth), [QFormer-13b](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/InstructBLIP/instruct_blip_vicuna13b_trimmed.pth) under `src/model_zoo/LAVIS`.

### Pretrained finetuned original models
The offered trained model weights can be found under the [table](https://github.com/dvlab-research/LLaMA-VID?tab=readme-ov-file#model) of the original author's repository. You can choose one of those models and download them under the `src/work_dirs`. If `work_dirs` folder is not created then you have to create it. 

### Pretrained finetuned extension models
The weights of the finetuned extension models can be found in the [LINK](NO LINK) and have to be downloaded in the `src/work_dirs`.

## Conda Environments
In this project, we had to create two different conda environments. Those are ***llamavid*** and ***llama_3_instruct***. This necessity initially arose to reproduce the performance of LLaMA_VID in the zero-shot video QA (Question Answering) benchmarks MSVD-QA and MSRVTT-QA. The authors utilize 'gpt-3.5-turbo' capabilities in text generation to evaluate the predictions made by the LLaMA-VID model in these Visual Question-answering tasks. Given an annotated pair of a question and a single-word answer related to the video, we need to evaluate if the prediction made by the proposed model agrees with the annotated single-word answer used as a label. Since 'gpt-3.5-turbo' is a closed model and we did not have access to an API key, we substituted it with[Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct), one of the newest open-source LLMs released by Meta that has text-generation capabilities. The caveat is that the latter LLM depends on a newer transformer version that is not compatible with the transformer version used in [LLAVA](https://arxiv.org/pdf/2304.08485), a backbone that LLaMA-VID builds upon, which justifies our decision to use two different environments. So whenever we need to evaluate the correctness of the LLaMA-VID model predictions in QA tasks, we use the *llama_3_instruct environment.

Both environments have a job script for building the environments in a cluster. To install them locally, one has to follow the following commands:

### llamavid installation
```
conda create -n llamavid python=3.10 -y
source activate llamavid

pip install --upgrade pip
pip install -e .

pip install ninja
pip install flash-attn==2.1.2.post3 --no-build-isolation
```
### llama_3_instruction installation
```
pip install --upgrade pip
pip install -r requirements_llama_three.txt
pip install openpyxl
pip install transformers==4.40.0
conda install -c conda-forge code-server
```

## How to reproduce the author's findings
The authors have used numerous benchmarks to have their findings. Precise steps have been given in the [evaluation](https://github.com/dvlab-research/LLaMA-VID?tab=readme-ov-file#evaluation) part of their repository. One has to execute the bash script job from `scripts/{image_only/video}/eval/{benchmark}.sh` by selecting either image_only or video which the benchmark is located and replace the name of the benchmark in the {benchmark} location.

<!-- ## How to reproduce the extension
THIS NEEDS INPUT!!! -->
