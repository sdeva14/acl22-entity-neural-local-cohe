# ACL22: Entity-based Neural Local Coherence Modeling
### [Sungho Jeon](https://sdeva14.github.io/) and [Michael Strube](https://www.h-its.org/people/prof-dr-michael-strube/)
#### [NLP Lab, Heidelberg Institute for Theoretical Studies (HITS)](https://www.h-its.org/research/nlp/people/)

This project contains a python implementation for the ACL22 paper whose title is "Entity-based Neural Local Coherence Modeling".

## Requirements

#### Conda environment
We recommend using Conda environment for a setup. It is easy to build an environment by the provided environment file. It is also possible to setup an environment manually by the information in "spec-file.txt". 

Our environment file is built based on CUDA9 driver and corresponding libraries, thus an environment should be managed by the target GPU environment. Otherwise, GPU flag should be disabled as a library. For the variation of XLNet, we use Transformers library implemented by Huggingface (Wolf et al, 2019).

    conda create --name py3_torch --file spec-file.txt
    source activate py3_torch
    pip install transformers==4.9.1
    pip install stanza==1.1.0

#### Dataset and materials
Datasets cannot be attached in the submission due to license problems, hence it should be downloaded from the given links.

- Dataset: The location of the target dataset should be configured in "build_config.py" with "--data_dir" option. The TOEFL dataset is available according to the link in the original paper (Blanchard et al. 2013) with LDC license. The index of the CV partition in TOEFL is attached as a file, whose name is ids_toefl_cv5.tar.gz. The NYT dataset can be downloaded with LDC license. We partition the dataset following previous work (Ferracaneet al. 2019), and attach our pre-processing script for this ("pp_nyt.py").

TOEFL dataset link: https://catalog.ldc.upenn.edu/LDC2014T06

NYT dataset link: https://catalog.ldc.upenn.edu/LDC2008T19

NYT partition link: https://github.com/elisaF/structured

GCDC dataset link: https://github.com/aylai/GCDC-corpus

For baselines which do not use a pretrained language model, we use the 100-dimensional pretrained embedding model on Google News, Glove (Pennington, Socher, and Manning 2014). We use the 50-dimensional embeddings on NYT. For our model and baselines employing the pretrained language model, we use the pretrained model "XLNet-base".

XLNet link: https://github.com/huggingface/transformers/

Glove link: https://nlp.stanford.edu/projects/glove/

## Run Models
#### Basic run
A basic run is performed by "main.py" with configuration options by providing in terminal or modifying "build_config.py" file.
Detail information about the configuration can be found in the "build_config.py"

	Examples for execution (assume that a data path is given in build_config.py).

    For TOEFL) python main.py --essay_prompt_id_train 1 --essay_prompt_id_test 1 --target_model acl22_entcoh

    For NYT) python main.py --target_model acl22_entcoh

#### The list of models in this framework
	conll17: The automated essay scoring model in Dong et al. (2017)
	emnlp18: The coherence model in Mesgar and Strube (2018)
	latent_doc_stru: The latent learning model in Liu and Lapata (2018)
	dis_avg: The first baseline which averages representations
	dis_tt: The second baseline which combines the averaged XLNet and the tree transformer
	
	emnlp19_unified: EMNLP19 model, Unified Coherence Model in Moon et al. (2019)
	cent_hds: Our original EMNLP20 model which approximates Centering theory
	cent_hds_np: Our EMNLP20 model but entity-based version
	cent_2enc_hds_np: Our EMNLP20 model but entity-based and encoding adjacent sentences
	avg_2enc: Simple baseline, averaging all sentence representations but encoding adjacent sentences
	acl22_entcoh: Our ACL22 model, Entity-based Cohernece Modeling

#### Pre-defined configuration
For convenient reproductions, we provide pre-defined configurations, configurations for RNN-based models (e.g., "toefl_build_config.py") and configurations for XLNet-based models (e.g., "toefl_xlnet_build_config.py") for the two datasets.
The location of the dataset and pretrained embedding layer should be managed properly in "build_config.py".

Note that additional parameters for baseline models should be configured as target models as described in the literatures

## Acknowledge
This implementation was possible thanks to many shared implementations. We describe an original source link at the first line of codes if we use theirs.
