## TASR: Timestep-Aware Diffusion Model for Image Super-Resolution

[Paper](https://arxiv.org/abs/2412.03355) 

<p align="center">
    <img src="assets/framework_v2.pdf">
</p>


<!-- ## <a name="visual_results"></a>Visual Results


[<img src="assets/visual_results/qual_cmp" height="223px"/>] -->



## <a name="installation"></a> Installation

```shell
# clone this repo
git clone https://github.com/SleepyLin/TASR.git
cd TASR
# create environment
conda create -n tasr python=3.10
conda activate tasr
pip install -r requirements.txt
```


## <a name="pretrained_models"></a>Pretrained Models
todo


## <a name="inference"></a>Inference
1. Download pretrained [Stable Diffusion v2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1-base) and [BSRNet](https://github.com/cszn/KAIR/releases/download/v1.0/BSRNet.pth)
2. Download pretrained [TASR v1 Model](https://huggingface.co/SleepyLin/TASR/blob/main/tasr_v1.pth)
3. Set correct model path in shell script
```shell
sh scripts/inference.sh
```


## <a name="Train"></a>Train
todo
Set ./config/train_tast.yaml
```shell
sh scripts/train.sh
```


## Citation

Please cite us if our work is useful for your research.

```
@misc{lin2024tasrtimestepawarediffusionmodel,
      title={TASR: Timestep-Aware Diffusion Model for Image Super-Resolution}, 
      author={Qinwei Lin and Xiaopeng Sun and Yu Gao and Yujie Zhong and Dengjie Li and Zheng Zhao and Haoqian Wang},
      year={2024},
      eprint={2412.03355},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.03355}, 
}
```


## Thanks

This project is based on [DiffBIR](https://github.com/XPixelGroup/DiffBIR). Thanks for their awesome work.

