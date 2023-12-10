

<div align="center">


<h2 align="center"> <a href="https://arxiv.org/abs/2307.08908">【ICCV'2023】What Can Simple Arithmetic Operations Do for Temporal Modeling?</a></h2>
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for latest update.  </h2>



[![Conference](http://img.shields.io/badge/ICCV-2023-f9f107.svg)](https://iccv2023.thecvf.com/)
[![Paper](https://img.shields.io/badge/Arxiv-2311.15732-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2307.08908)


[Wenhao Wu](https://whwu95.github.io/)<sup>1,2</sup>, [Yuxin Song]()<sup>2</sup>, [Zhun Sun]()<sup>2</sup>, [Jingdong Wang](https://jingdongwang2017.github.io/)<sup>3</sup>, [Chang Xu](http://changxu.xyz/)<sup>1</sup>, [Wanli Ouyang](https://wlouyang.github.io/)<sup>3,1</sup>

 
<sup>1</sup>[The University of Sydney](https://www.sydney.edu.au/), <sup>2</sup>[Baidu](https://vis.baidu.com/#/), <sup>3</sup>[Shanghai AI Lab](https://www.shlab.org.cn/)

</div>


***
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/what-can-simple-arithmetic-operations-do-for/action-recognition-in-videos-on-something-1)](https://paperswithcode.com/sota/action-recognition-in-videos-on-something-1?p=what-can-simple-arithmetic-operations-do-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/what-can-simple-arithmetic-operations-do-for/action-classification-on-kinetics-400)](https://paperswithcode.com/sota/action-classification-on-kinetics-400?p=what-can-simple-arithmetic-operations-do-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/what-can-simple-arithmetic-operations-do-for/action-recognition-in-videos-on-something)](https://paperswithcode.com/sota/action-recognition-in-videos-on-something?p=what-can-simple-arithmetic-operations-do-for)

This is the official implementation of our **ATM** (Arithmetic Temporal Module), which explores the potential of four simple arithmetic operations for temporal modeling. 

Our best model can achieve **89.4%** Top-1 Acc. on Kinetics-400, **65.6%** Top-1 Acc. on Something-Something V1, **74.6%** Top-1 Acc. on Something-Something V2!


<details open><summary>🔥 I also have other recent video recognition projects that may interest you ✨. </summary><p>

> [**Side4Video: Spatial-Temporal Side Network for Memory-Efficient Image-to-Video Transfer Learning**](https://arxiv.org/abs/2311.15769)<br>
> Huanjin Yao, Wenhao Wu, Zhiheng Li<br>
> [![arXiv](https://img.shields.io/badge/Arxiv-2311.15769-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2311.15769) [![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/HJYao00/Side4Video) 



> [**Bidirectional Cross-Modal Knowledge Exploration for Video Recognition with Pre-trained Vision-Language Models**](https://arxiv.org/abs/2301.00182)<br>
> Wenhao Wu, Xiaohan Wang, Haipeng Luo, Jingdong Wang, Yi Yang, Wanli Ouyang <br>
> [![Conference](http://img.shields.io/badge/CVPR-2023-f9f107.svg)](https://openaccess.thecvf.com/content/CVPR2023/html/Wu_Bidirectional_Cross-Modal_Knowledge_Exploration_for_Video_Recognition_With_Pre-Trained_Vision-Language_CVPR_2023_paper.html) [![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/whwu95/BIKE) 


> [**Revisiting Classifier: Transferring Vision-Language Models for Video Recognition**](https://arxiv.org/abs/2207.01297)<br>
> Wenhao Wu, Zhun Sun, Wanli Ouyang <br>
> [![Conference](http://img.shields.io/badge/AAAI-2023-f9f107.svg)](https://ojs.aaai.org/index.php/AAAI/article/view/25386/25158) [![Journal](http://img.shields.io/badge/IJCV-2023-Bf107.svg)](https://link.springer.com/article/10.1007/s11263-023-01876-w) [![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/whwu95/Text4Vis) 





</p></details>


<!-- ## Content
- [Content](#content)
- [📣 News](#-news)
- [🌈 Overview](#-overview)
- [📌 BibTeX \& Citation](#-bibtex--citation)
- [🎗️ Acknowledgement](#️-acknowledgement)
- [👫 Contact](#-contact) -->



## 📣 News

<!-- - [ ] `TODO`: All models will be released. -->
- [x] `Nov 29, 2023`: Training codes have be released.
- [x] `July 14, 2023`: 🎉Our **ATM** has been accepted by **ICCV-2023**.



## 🌈 Overview
![ATM](pics/ATM.png)
The key motivation behind ATM is to explore the potential of simple arithmetic operations to capture auxiliary temporal clues that may be embedded in current video features, without relying on the elaborate design. The ATM can be integrated into both vanilla CNN backbone (e.g., ResNet) and Vision Transformer (e.g., ViT) for video action recognition.


## 🚀 Training & Testing
We offer training and testing scripts for Kinetics-400, Sth-Sth V1, and Sth-Sth V2. Please refer to the [*script*](https://github.com/whwu95/ATM/tree/main/scripts) folder for details. For example, you can run:

```sh
# Train the 8 Frames ViT-B/32 model on Sth-Sth v1.
sh scripts/ssv1/train_base.sh 

# Test the 8 Frames ViT-B/32 model on Sth-Sth v1.
sh scripts/ssv1/test_base_f8.sh
```



<a name="bibtex"></a>
## 📌 BibTeX & Citation

If you use our code in your research or wish to refer to the baseline results, please use the following BibTeX entry😁.


```bibtex
@inproceedings{atm,
  title={What Can Simple Arithmetic Operations Do for Temporal Modeling?},
  author={Wu, Wenhao and Song, Yuxin and Sun, Zhun and Wang, Jingdong and Xu, Chang and Ouyang, Wanli},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  year={2023}
}
```


<a name="acknowledgment"></a>
## 🎗️ Acknowledgement

This repository is built upon portions of [VideoMAE](https://github.com/MCG-NJU/VideoMAE), [CLIP](https://github.com/openai/CLIP), and [EVA](https://github.com/baaivision/EVA). Thanks to the contributors of these great codebases.


## 👫 Contact
For any question, please file an issue or contact [Wenhao Wu](https://whwu95.github.io/).
