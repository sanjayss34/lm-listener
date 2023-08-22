# Can Language Models Learn to Listen?
This is the repo for the paper [Can Language Models Learn to Listen?](https://arxiv.org/abs/2308.10897), appearing at ICCV 2023.

## Setup Environment
Create a new Python 3 environment and install PyTorch 1.11.0 from https://pytorch.org/get-started/previous-versions/. Then install the requirements for this repo via `pip install -r requirements.txt`.
Also, please clone the DECA (for visualization) and EMOCA (for emotion/valence evaluation) repositories, and set the following environment variables:
```
export PYTHONPATH=/PATH/TO/EMOCA/:$PYTHONPATH
export DECA_PATH=/PATH/TO/DECA/
```
You will need to change EMOCA emotion recogition to not process from image. In `gdl/models/EmoDeca.py`, add the following lines to the beginning of the `forward` method:
```
if 'image' not in batch:
    values = batch
else:
    values = self.deca.encode(batch, training=False)
```
You will also need to download the DECA and EMOCA models (there are instructions in those repos).

## Data Preparation
Please download the data from the Google Driver folder: [here](https://drive.google.com/file/d/1fR4sobslLB0gESQj6zya63XgupJFpVjc/view?usp=sharing). Place the data so that there are directories `dataset/trevor`, `dataset/conan`, `dataset/stephen`, and `dataset/trevorconanstephen` that have the corresponding segment files.
Note: If you want to use a cross-speaker VQ to train an LM Listener for a speaker (as we did for Conan and Stephen), you should copy the corresponding speaker's directory and then overwrite the `mean.npy` and `std.npy` files with the files from the `trevorconanstephen` directory. For instance, for Conan, you should copy `dataset/conan` to `dataset/conanglobal` and then copy `dataset/trevorconanstephen/{mean,std}.npy` to `dataset/conanglobal/`.

## Pre-trained model
We provide a pre-trained VQ model and LM Listener for Trevor Noah [here](https://drive.google.com/drive/folders/1WMAsrky61gI36x_IstkoNzuNiqCmhKJV?usp=sharing).

## Training
The following command will train a VQ encoder-decoder:
```
python3 train_vq.py \
	--batch-size 256 \
	--lr 2e-4 \
	--total-iter 300000 \
	--lr-scheduler 200000 \
	--nb-code 256 \
	--down-t 3 \
	--depth 3 \
	--window-size 32 \
	--dilation-growth-rate 3 \
	--out-dir output \
	--dataname face_{trevor/trevorconanstephen} \
	--vq-act relu \
	--quantizer ema_reset \
	--loss-vel 0.5 \
	--recons-loss l1_smooth \
	--exp-name VQVAE_{trevor/trevorconanstephen}
```
The following command will train an LM Listener:
```
python train_t2m_trans.py \
	--exp-name listener_{trevor/conanglobal/stephenglobal} \
	--batch-size 8 \
	--nb-code 256 \
	--drop-out-rate 0.1 \
	--resume-pth output/VQVAE_{trevor/trevorconanstephen}/net_iter300000.pth \
	--vq-name VQVAE_{trevor/trevorconanstephen} \
	--out-dir output \
	--total-iter 100000 \
	--lr-scheduler 150000 \
	--lr 0.00005 \
	--dataname face_realtalkv2 \
	--down-t 2 \
	--depth 3 \
	--quantizer ema_reset \
	--eval-iter 2000 \
	--pkeep 0.50 \
	--dilation-growth-rate 3 \
	--vq-act relu \
	--max-motion-length 240 \
	--gpt2 gpt2-medium \
	--print_val_pred \
	--gradient_accumulation_steps 2 \
	--manual-bf16 \
	--delay-start-frames 96 \
	--max-time-before 3
```

## Generation
The following command can be used to generate prediction files (in `.npy` format) from a trained LM Listener:
```
python train_t2m_trans.py \
        --exp-name listener_{trevor/conanglobal/stephenglobal} \
        --batch-size 8 \
        --nb-code 256 \
        --drop-out-rate 0.1 \
        --resume-pth output/VQVAE_{trevor/trevorconanstephen}/net_iter300000.pth \
	--vq-name VQVAE_{trevor/trevorconanstephen} \
        --out-dir output \
        --total-iter 0 \
        --lr-scheduler 150000 \
        --lr 0.00005 \
        --dataname face_trevor \
        --down-t 3 \
        --depth 3 \
        --quantizer ema_reset \
        --eval-iter 2000 \
        --pkeep 0.50 \
        --dilation-growth-rate 3 \
        --vq-act relu \
        --max-motion-length 240 \
        --gpt2 gpt2-medium \
        --print_val_pred \
        --gradient_accumulation_steps 2 \
        --manual-bf16 \
        --delay-start-frames 96 \
        --max-time-before 3 \
        --save-name subdir_where_predictions_will_be_saved \
        --seed 50 \
        --resume-trans /path/to/model/checkpoint.pth
```

## Evaluation
The following command can be used to compute evaluation metrics for an LM Listener:
```
python evaluate_listener.py --output_dir output/{EXPERIMENT_NAME} --segments_path dataset/{trevor/conanglobal/stephenglobal}/segments_val.pth --mean_std_path dataset/{trevor/conanglobal/stephenglobal}/
```

## Baselines
To produce a directory of predictions for the Random VQ, Random Train, and Nearest Neighbor baselines, use the following command templates:
```
python random_baseline.py --vq-dir dataset/{trevor/conanglobal/stephenglobal}/vqvae_{trevor/trevorconanstephen}_val/ --output-dir output/{trevor/conan/stephen}_random_vq --params-path path_to_vq_config.json --max-motion-length 240 --history-size 3 --mean-std-path dataset/{trevor/conanglobal/stephenglobal}/
python random_baseline.py --vq-dir dataset/{trevor/conanglobal/stephenglobal}/vqvae_{trevor/trevorconanstephen}_val/ --output-dir output/{trevor/conan/stephen}_nearest_neighbor --params-path path_to_vq_config.json --max-motion-length 240 --history-size 3 --mean-std-path dataset/{trevor/conanglobal/stephenglobal}/ --train-segments-path dataset/{trevor/conanglobal/stephenglobal}/segments_train.pth --val-segments-path dataset/{trevor/conanglobal/stephenglobal}/segments_val.pth --nearest-neighbor --embedding-model-name sentence-transformers/all-mpnet-base-v2 --batch-size 32 --normalize
python random_baseline.py --vq-dir dataset/{trevor/conanglobal/stephenglobal}/vqvae_{trevor/trevorconanstephen}_val/ --output-dir output/{trevor/conan/stephen}_random_train --params-path path_to_vq_config.json --max-motion-length 240 --history-size 3 --mean-std-path dataset/{trevor/conanglobal/stephenglobal}/ --train-segments-path dataset/{trevor/conanglobal/stephenglobal}/segments_train.pth --val-segments-path dataset/{trevor/conanglobal/stephenglobal}/segments_val.pth
```
The format of the predictions is `.npy`, just like the predictions produced by the LM Listener.

## Visualization
The following command can be used to generate visualizations for an LM Listener:
```
python visualize_listener.py --output_dir /path/to/output/dir/ --segments_path dataset/{trevor/conanglobal/stephenglobal}/segments_val.pth --default_code_path default_code_trevor_emoca2.pkl --params_path output/{EXPERIMENT_NAME}/config.json  --items output/{EXPERIMENT_NAME}/,vq,gt,video --mean_std_path dataset/{trevor/conanglobal/stephenglobal}/ --audio_root /path/to/raw/audios/ --video_root /path/to/raw/videos/ --fps 30
```
The `--items` parameter allows you to specify a comma-separated list of what to visualize. The options are: `video` (raw video), `gt` (the ground-truth EMOCA face reconstruction of the listener), `vq` (the VQ reconstruction of the listener), or a path to the output directory containing the predicted `.npy` files of an LM Listener.

## Acknowledgements
Much of the code in this repo is taken from [T2M-GPT](https://github.com/Mael-zys/T2M-GPT).

## Citation
```
@inproceedings{ng2023text2listen,
   title={Can Language Models Learn to Listen?}
   author={Ng, Evonne and Subramanian, Sanjay
            and Klein, Dan and Kanazawa, Angjoo
            and Darrell, Trevor and Ginosar, Shiry},
   booktitle={Proceedings of the International
            Conference on Computer Vision (ICCV)},
   year={2023}
}
```
