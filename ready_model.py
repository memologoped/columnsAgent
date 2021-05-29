from os.path import join

import torch

import config
from models.attention_model import ZReader as ZReader_attn
from models.fnet_model import ZReader as ZReader_fnet

if __name__ == "__main__":
    str_true = "thehighwayroundsacurveandtransitionstondstreetrunningeastwardthroughmorefarmlandasthetrunklineappro" \
               "achesneway"
    str_ = "twer hvb rye hfj idf g fdhh ghw kja ghjy r rtyo u nfgh dhjk s a cghfhf u r vfgh e a fn d t r afgh n s i " \
           "tfgh i ghjo n srt t o nghj d smn t rkl e edfg t fdr u fdn n iret hn rtyg e adfsg s t wdfg a r vbd t " \
           "hvbcnv r o u iopg xcvh zxm sdo qwr dfge frety a dfgr m l kla ern drt auio jks vbnt bvnh e fght r u dsfn" \
           " ikk bnl gfi kbn fe ea hgp dsfp feir bnco ajkl etc dfh ehjd s dgn e dfw dfka yghp"
    batch_ = [str_]

    with open(file="ztext.txt", encoding="utf-8", mode="r") as file:
        batch = file.readlines()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ZReader_attn(*ZReader_attn.get_parameters()).to(device)
    model.load_parameters(join(config.weights_path, '0529_1629_72'), device=device)
    model.z_read(batch_)
