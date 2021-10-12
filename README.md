## Tripartite Information Mining and Integration for Image Matting (Timi-Net)

*Yuhao Liu\*, Jiake Xie\*, Xiao Shi, Yu Qiao, Yujie Huang, Yong Tang, Xin Yang*
This is the official PyTorch implementation of our paper [**Tripartite Information Mining and Integration for Image Matting**](https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_Tripartite_Information_Mining_and_Integration_for_Image_Matting_ICCV_2021_paper.pdf) that has been accepted to 2021 IEEE/CVF International Conference on Computer Vision (ICCV 2021).
***

## Get Started

## Requirements ##

The repository is trained and tested on Ubuntu 18.04.3 LTS, based on the main following settings.

- [x] python 3.6+  
- [x] Pytorch 1.1.0+
- [x] cuda 11.0+

## Usage ##

#### Testing

To quickly test sample images with our model (trained on Adobe dataset), you can just run through

```shell
cd Timi-Net
python test.py 
```

By default, the code takes the data in the "./inputs/" folder, loads the "TIMI-Net.pth" model and saves results in the "./outputs/" folder.  Please read the code to see other parameter settings. 

<h4>Results AND Model</h4>

| Datasets             |                           Results                            |                            Model                             |
| -------------------- | :----------------------------------------------------------: | :----------------------------------------------------------: |
| Adobe Composition-1K | <a herf='https://drive.google.com/drive/folders/1XBCyY2TVAxTPLt0EvP9525pp32QKWFm0'>Google Drive</a>/<a herf='https://pan.baidu.com/s/1U27EFW17yXpBSHbmNCM_Cg'>Baidu Drive</a>(code: jhmr) | <a herf='https://drive.google.com/file/d/1c1gdvhXG-PqoZFJsL1HIidomN88i_PCv/view'>Google Drive</a> |
| Distinctions-646     |                          is coming                           |                          is coming                           |
| Human-2K             |                          is coming                           |                          is coming                           |



<h3><strong><i>🚀 Training code is coming soon...</i></strong></h3>



## Statement

This project is only for research purpose. For any other questions, please feel free to contact us.

## Related Projects

This repository highly depends on the **GCA-matting** repository at https://github.com/Yaoyi-Li/GCA-Matting. We thank the authors of GCA for their great work and clean code.

## BibTex
If you use this code for your research, please cite our paper.

 ```tex
@InProceedings{Liu_2021_ICCV,
    author    = {Liu, Yuhao and Xie, Jiake and Shi, Xiao and Qiao, Yu and Huang, Yujie and Tang, Yong and Yang, Xin},
    title     = {Tripartite Information Mining and Integration for Image Matting},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {7555-7564}
}
 ```
