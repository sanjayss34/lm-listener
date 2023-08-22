import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(description='Optimal Transport AutoEncoder training for Amass',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    ## dataloader
    
    parser.add_argument('--dataname', type=str, default='kit', help='dataset directory')
    parser.add_argument('--batch-size', default=128, type=int, help='batch size')
    parser.add_argument('--fps', default=[30], nargs="+", type=int, help='frames per second')
    parser.add_argument('--seq-len', type=int, default=64, help='training motion length')
    parser.add_argument('--max-motion-length', type=int, default=128, help='max motion length')
    parser.add_argument('--max-tokens', type=int, default=None)
    parser.add_argument('--step-size', type=int, default=None, help='max motion length')
    parser.add_argument('--train_eval', action="store_true")
    parser.add_argument('--test-eval', action='store_true')
    parser.add_argument('--data_v2', action="store_true")
    parser.add_argument('--no-before-text', action="store_true")
    parser.add_argument('--max-time-before', type=int, default=None)
    parser.add_argument('--normalize-speaker', action="store_true")
    parser.add_argument('--normalize-audio', action="store_true")
    parser.add_argument('--delay-start-frames', type=int, default=0)
    parser.add_argument('--train-min-length', type=int, default=0)
    parser.add_argument('--val-min-length', type=int, default=0)
    
    ## optimization
    parser.add_argument('--total-iter', default=100000, type=int, help='number of total iterations to run')
    parser.add_argument('--warm-up-iter', default=1000, type=int, help='number of total iterations for warmup')
    parser.add_argument('--lr', default=2e-4, type=float, help='max learning rate')
    parser.add_argument('--lr-scheduler', default=[60000], nargs="+", type=int, help="learning rate schedule (iterations)")
    parser.add_argument('--gamma', default=0.05, type=float, help="learning rate decay")
    
    parser.add_argument('--weight-decay', default=1e-6, type=float, help='weight decay') 
    parser.add_argument('--decay-option',default='all', type=str, choices=['all', 'noVQ'], help='disable weight decay on codebook')
    parser.add_argument('--optimizer',default='adamw', type=str, choices=['adam', 'adamw'], help='disable weight decay on codebook')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--grad-scaling', action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--fp16-half", action="store_true")
    parser.add_argument("--manual-bf16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--train-loss-threshold", type=float, default=0.001)
    parser.add_argument("--training-end-check-interval", type=int, default=600)
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--linear-scheduler", action="store_true")
    
    ## vqvae arch
    parser.add_argument("--code-dim", type=int, default=512, help="embedding dimension")
    parser.add_argument("--nb-code", type=int, default=512, help="nb of embedding")
    parser.add_argument("--mu", type=float, default=0.99, help="exponential moving average to update the codebook")
    parser.add_argument("--down-t", type=int, default=3, help="downsampling rate")
    parser.add_argument("--stride-t", type=int, default=2, help="stride size")
    parser.add_argument("--width", type=int, default=512, help="width of the network")
    parser.add_argument("--depth", type=int, default=3, help="depth of the network")
    parser.add_argument("--dilation-growth-rate", type=int, default=3, help="dilation growth rate")
    parser.add_argument("--output-emb-width", type=int, default=512, help="output embedding width")
    parser.add_argument('--vq-act', type=str, default='relu', choices = ['relu', 'silu', 'gelu'], help='dataset directory')

    ## gpt arch
    parser.add_argument("--block-size", type=int, default=25, help="seq len")
    parser.add_argument("--embed-dim-gpt", type=int, default=512, help="embedding dimension")
    parser.add_argument("--clip-dim", type=int, default=512, help="latent dimension in the clip feature")
    parser.add_argument("--num-layers", type=int, default=2, help="nb of transformer layers")
    parser.add_argument("--n-head-gpt", type=int, default=8, help="nb of heads")
    parser.add_argument("--ff-rate", type=int, default=4, help="feedforward size")
    parser.add_argument("--drop-out-rate", type=float, default=0.1, help="dropout ratio in the pos encoding")
    parser.add_argument("--text-model-name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--extra_input_dim", type=int, nargs='+', default=[])
    parser.add_argument("--include-speaker", action="store_true")
    parser.add_argument("--include-audio", action="store_true")
    parser.add_argument("--include-speaker-before", action="store_true")
    parser.add_argument("--include-audio-before", action="store_true")
    parser.add_argument("--text_token_level", action="store_true")
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--no-text", action="store_true")
    parser.add_argument("--no-end", action="store_true")
    parser.add_argument("--gpt2", type=str, default=None)
    parser.add_argument("--sentiment-token", type=str, default=None)
    parser.add_argument("--freeze-lm", action="store_true")
    parser.add_argument("--num-output-layers", type=int, default=0)
    parser.add_argument("--transformer-not-pretrained", action="store_true")
    parser.add_argument("--speaker-pkeep", type=float, default=None)
    parser.add_argument("--audio-pkeep", type=float, default=None)
    parser.add_argument("--speaker-vq-path")
    parser.add_argument("--speaker-vq-loss", action="store_true")
    parser.add_argument("--audio-vq-path")
    parser.add_argument("--audio-vq-loss", action="store_true")
    parser.add_argument("--fix-pkeep", action="store_true")
    parser.add_argument("--fixed-text-token", action="store_true")
    parser.add_argument("--fixed-text-token-not-space", action="store_true")
    parser.add_argument("--fixed-text-token-not-punctuation", action="store_true")
    parser.add_argument("--unaligned-text", action="store_true")
    parser.add_argument("--remove-space-before-vq-tokens", action="store_true")
    parser.add_argument("--random-text-token-order", action="store_true")
    
    ## quantizer
    parser.add_argument("--quantizer", type=str, default='ema_reset', choices = ['ema', 'orig', 'ema_reset', 'reset'], help="eps for optimal transport")
    parser.add_argument('--quantbeta', type=float, default=1.0, help='dataset directory')

    ## resume
    parser.add_argument("--resume-pth", type=str, default=None, help='resume vq pth')
    parser.add_argument("--resume-trans", type=str, default=None, help='resume gpt pth')
    
    
    ## output directory 
    parser.add_argument('--out-dir', type=str, default='output_GPT_Final/', help='output directory')
    parser.add_argument('--exp-name', type=str, default='exp_debug', help='name of the experiment, will create a file inside out-dir')
    parser.add_argument('--vq-name', type=str, default='exp_debug', help='name of the generated dataset .npy, will create a file inside out-dir')
    ## other
    parser.add_argument('--print-iter', default=200, type=int, help='print frequency')
    parser.add_argument('--eval-iter', default=5000, type=int, help='evaluation frequency')
    parser.add_argument('--seed', default=123, type=int, help='seed for initializing training. ')
    parser.add_argument("--if-maxtest", action='store_true', help="test in max")
    parser.add_argument('--pkeep', type=float, default=1.0, help='keep rate for gpt training')
    parser.add_argument('--print_val_pred', action='store_true')
    parser.add_argument('--save-name', default=None)
    parser.add_argument('--num-samples', default=1, type=int)

    parser.add_argument("--valence-window-size", type=int, default=90)
    parser.add_argument('--control-sentiment', action="store_true")
    
    
    return parser.parse_args()
