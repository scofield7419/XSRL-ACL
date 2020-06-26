
## Translation-based projection ##

- Step1. Train POS taggers for each languages based on the corresponding labelled POS data (with universal POS tagset).
    - `trainPOS.py` (config the `Config/config.cfg` in advance)

- Step2. Prepare the translated SRL parallel data.
    - The translation process should be fulfilled by you own. You can achieve it via [*Google translation API*](https://translate.google.com).
    - The format of source side data and translated target side data should follow the example data in `Data` fold: `en-de-train-src.conllu` and `en-de-train-tgt.conllu`. 
        It's *conllu* style. 
    - We provide the *UPB-English* data which was not offered by [UPBV1.0](https://github.com/System-T/UniversalPropositions) at the time (i.e., 2019/10/1) we perform our experiments.
        
- Step3. Generating the aligning file for each pair of source and target language dataset.
    - Install the [`fast_align`](https://github.com/clab/fast_align), and conducting alignment.
    - The format of the alignment output file follows the example file `en-de-train-src2tgt-align.prob` in `Data/upb_parallel/en-de` fold. 

- Step4. Start annotation projection.
    - `project.py` 

----

#### Note:

- *Please note that one sentence may contain multiple sets of prd-args structure.
        So you should pre-process the data and split them in advance, 
        making sure that one sentence in the data only at maximum contains one set of prd-args proposition.*

