# Co-training algorithm on Educational Data Mining 
An implementation of the *Co-training* scheme, a well-known multi-view Semi-Supervised Learning approach, applied on Educational Data Mining datasets related with the task of _Early Prognosis of Academic Performance_ is provided here. A specific split of the feature vector is applied each time based on the two separate views that exist into the original data.

Apart from the proposed algorithm **Cotrain(Extra, GBC)**, the results of several other variant of Co-training scheme are provided, as well as the results of *Self-training* approaches. Moreover, the results of the *CoForest* algorithm (implemented by  *Mr. Ming Li* lim@lamda.nju.edu.cn - [link](http://lamda.nju.edu.cn/code_CoForest.ashx)) have been computed for the same datasets' splits and same seeds.


## Datasets

A full description is provided on the related publication.
More comments are going to be posted, after the acceptance of the submission.

## Citation

Please cite the following paper if you use our algorithm/datasets for your work.

'''

Submitted on http://ieee-edusociety.org/about/about-ieee-transactions-learning-technologies

IEEE repository: https://ieeexplore.ieee.org/document/8692618

Georgios Kostopoulos, Stamatis Karlos, Sotiris Kotsiantis:
Multiview Learning for Early Prognosis of Academic Performance: A Case Study. TLT 12(2): 212-224 (2019)

'''

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
## Notes

More information about the authors are provided in [ml.math.upatras.gr](http://ml.math.upatras.gr/)
