# Co-training algorithm on Educational Data Mining 

![EDM_cotraining_splits](https://user-images.githubusercontent.com/6009931/103440685-5d385680-4c50-11eb-874c-7c2bdcbae785.png)

                                                                       
An implementation of the *Co-training* scheme, a well-known multi-view Semi-Supervised Learning approach, applied on Educational Data Mining (**EDM**) datasets related with the task of _Early Prognosis of Academic Performance_ is provided here. A specific split of the feature vector is applied each time based on the two separate views that exist into the original data. Thus, we are not based on random feature split, but we examine the efficacy of fitting a *Co-training* scheme on two independent views (verified by correlation test on our datasets). 

Apart from the proposed algorithm **Cotrain(Extra, GBC)**, the results of several other variants of *Co-training* scheme are provided, as well as the results of *Self-training* approaches. Moreover, the results of the *CoForest* algorithm (implemented by  *Mr. Ming Li* lim@lamda.nju.edu.cn - [link](http://lamda.nju.edu.cn/code_CoForest.ashx)) have been computed for the same datasets' splits. Fixed number of seeds are provided for verifying the results that are demonstrated into the original publication.


## Datasets

A full description is provided on the related publication. However, we still provide here some information.

The two separate feature views describe:
- (V1) students' characteristics and academic achievements (mixed features),
- (V2) time-variant attributes regarding students' online activity (numerical features).


Our binary dataset consists of 1073 instances, where 4 variants of it are created based on the selected feature vector. This last property depends on the time intervals during which new attributes are produced (e.g. semesters).  



## Citation

Please cite the following paper if you use our algorithm/datasets for your work.


      @article{DBLP:journals/tlt/KostopoulosKK19,
        author    = {Georgios Kostopoulos and
                     Stamatis Karlos and
                     Sotiris Kotsiantis},
        title     = {Multiview Learning for Early Prognosis of Academic Performance: {A}
                     Case Study},
        journal   = {{IEEE} Trans. Learn. Technol.},
        volume    = {12},
        number    = {2},
        pages     = {212--224},
        year      = {2019},
        url       = {https://doi.org/10.1109/TLT.2019.2911581},
        doi       = {10.1109/TLT.2019.2911581},
        timestamp = {Fri, 03 Apr 2020 13:47:12 +0200},
        biburl    = {https://dblp.org/rec/journals/tlt/KostopoulosKK19.bib},
        bibsource = {dblp computer science bibliography, https://dblp.org}
      }

Submitted on http://ieee-edusociety.org/about/about-ieee-transactions-learning-technologies

IEEE repository: https://ieeexplore.ieee.org/document/8692618

Georgios Kostopoulos, Stamatis Karlos, Sotiris Kotsiantis:
Multiview Learning for Early Prognosis of Academic Performance: A Case Study. TLT 12(2): 212-224 (2019)


## Basic Dependencies

* Python 2.7

* Python dependencies for main algorithm
```
pip install -r requirements.txt

```
* Python 3.x

* Python dependencies for visualizations algorithm
```
pip install -r requirements_draw.txt

```
## Notes - Members

More information about the authors are provided in [ml.math.upatras.gr](http://ml.math.upatras.gr/).

- George Kostopoulos: kostg@sch.gr
- Stamatis Karlos: stkarlos@upatras.gr
- Sotiris Kotsiantis: sotos@math.upatras.gr
