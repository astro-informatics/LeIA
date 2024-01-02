# Learned Image reconstruction in Astronomy (LeIA)

In this repo you can find the code used to produce the results in the paper "Learned interferometric imaging for the SPIDER instrument". It presents two different learned approaches for interferometric imaging problems in astronomy. For more information about the methods implemented in this repo, please refer to the [paper](https://academic.oup.com/rasti/article-pdf/2/1/760/54877062/rzad054.pdf).

The script used to train the networks can be found in `src/examples/train_SPIDER.py` and the config files in `configs/`. I am currently cleaning up the code and improving documentation, so if you have any questions feel free to get in touch at `matthijs.mars.20 [at] ucl.ac.uk`.

This repository contains the code used to produce the results in the paper "Learned interferometric imaging for the SPIDER instrument". The script used to train the networks can be found in `src/examples/train_SPIDER.py` and the config files in `configs/`. I am currently cleaning up the code and improving documentation. 

If you make use of any of this code, please cite the paper:

``` 
@article{10.1093/rasti/rzad054,
    author = {Mars, Matthijs and Betcke, Marta M and McEwen, Jason D},
    title = "{Learned interferometric imaging for the SPIDER instrument}",
    journal = {RAS Techniques and Instruments},
    volume = {2},
    number = {1},
    pages = {760-778},
    year = {2023},
    month = {11},
    issn = {2752-8200},
    doi = {10.1093/rasti/rzad054},
    url = {https://doi.org/10.1093/rasti/rzad054},
    eprint = {https://academic.oup.com/rasti/article-pdf/2/1/760/54877062/rzad054.pdf},
}
```