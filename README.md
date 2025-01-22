# SoccerNet-v2 - Action Spotting

This repository contains the code and paper submitted my group for the final project in Georgia Tech's Graduate Deep Learning class. For our project we tackled the SoccerNet action spotting challenge in which submissions must localize when and which soccer action occurs, in a dataset among 17 classes. Each action is annotated with a single timestamp, making those annotations quite scattered in long videos containing full soccer broadcasts.

We utilized the work of Cioppa et al for the paper "A Context-Aware Loss Function for Action Spotting in Soccer Videos" as a starting point for our work and we set out to beat the benchmark of their approach. Notable changes are made to the code and logic for the model, training loop, and loss functions. My primary contribution to the effort came from designing and coding our model architecture and conducting experiments measuring performance. A number of approaches were attempted but our best results which successfully beat our targeted baseline came from incorporating multihead attention after initial feature extracting CNN layers in conjunction from other functional changes like additional regularization and adjustments to spotting logic. Expanded elaboration on this can be found in our paper which is stored in this repo.


```bibtex
@InProceedings{Deliege2020SoccerNetv2,
  author = { Deliège, Adrien and Cioppa, Anthony and Giancola, Silvio and Seikavandi, Meisam J. and Dueholm, Jacob V. and Nasrollahi, Kamal and Ghanem, Bernard and Moeslund, Thomas B. and Van Droogenbroeck, Marc},
  title = {SoccerNet-v2 : A Dataset and Benchmarks for Holistic Understanding of Broadcast Soccer Videos},
  booktitle = {CoRR},
  month = {Nov},
  year = {2020}
}
```


```bibtex
@InProceedings{Cioppa2020Context,
  author = {Cioppa, Anthony and Deliège, Adrien and Giancola, Silvio and Ghanem, Bernard and Van Droogenbroeck, Marc and Gade, Rikke and Moeslund, Thomas B.},
  title = {A Context-Aware Loss Function for Action Spotting in Soccer Videos},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2020}
}
```

