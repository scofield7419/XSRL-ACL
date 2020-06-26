codes for the ACL 2020 paper: [Cross-Lingual Semantic Role Labeling with High-Quality Translated Training Corpus](https://www.aclweb.org/anthology/2020.acl-main.627/)


### Cite:

```buildoutcfg
@inproceedings{fei-etal-2020-cross,
    title = "Cross-Lingual Semantic Role Labeling with High-Quality Translated Training Corpus",
    author = "Fei, Hao  and
      Zhang, Meishan  and
      Ji, Donghong",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    url = "https://www.aclweb.org/anthology/2020.acl-main.627",
    pages = "7014--7026",
}
```


---


## A. Translation-based projection ##

- Step1. Train POS taggers for each languages based on the corresponding labelled POS data (with universal POS tagset).
    - `trainPOS.py` (config the `Config/config.cfg` in advance)

- Step2. Prepare the translated SRL parallel data.
    - The translation process should be fulfilled by you own. You can achieve it via [*Google translation API*](https://translate.google.com).
    - The format of source side data and translated target side data should follow the example data in `Data` fold. 
        It's *conllu* style. 
    - We provide the *UPB-English* data which was not offered by [UPBV1.0](https://github.com/System-T/UniversalPropositions) at the time (i.e., 2019/10/1) we perform our experiments.
        
- Step3. Generating the aligning file for each pair of source and target language dataset.
    - Install the [`fast_align`](https://github.com/clab/fast_align), and conducting alignment.
    - The format of the alignment output file follows the example file `src2tgt-align.prob` in `Data` fold. 

- Step4. Start annotation projection.
    - `project.py` 


#### Note:

- *Please note that one sentence may contain multiple sets of prd-args structure.
        So you should pre-process the data and split them in advance, 
        making sure that one sentence in the data only at maximum contains one set of prd-args proposition.*



----

## B. Code for PGN-LSTM SRL model

- step 1. Configure the setting file *Config/config.cfg*

- step 2. Run with *main.py*


----

#### Note:

- Environment dependency:
    - python3.7
    - pytorch1.0.1
    - allennlp
    - huggingface-bert
    - tqdm
 

- Download the [UPBV1.0 data](https://github.com/System-T/UniversalPropositions).
    Please pre-process the data, making sure that one sentence in the data only at maximum contains one set of prd-args proposition.*

