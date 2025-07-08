out_dir=""

# Check if the first argument is --out-dir and the second is non-empty
if [[ "$1" == "--out-dir" && -n "$2" ]]; then
    out_dir="$2"
    shift 2
elif [[ "$1" == "--out-dir" && -z "$2" ]]; then
    echo "Error: --out-dir requires a directory argument."
    exit 1
fi

# Build the argument string for output directory
OUT_ARG=""
if [ -n "$out_dir" ]; then
    OUT_ARG="--out-dir $out_dir"
fi

### training CB-AE for CelebA-HQ-pretrained StyleGAN2 with supervised models as pseudolabeler
python -u train/train_cbae_gan.py -d celebahq40 -e cbae_stygan2_thr90 -p supervised -s 40concepts $OUT_ARG

### training CC for CelebA-HQ-pretrained StyleGAN2 with supervised models as pseudolabeler
# python3 -u train/train_cc_gan.py -e cc_stygan2_thr90 -p supervised -t sup_pl_cls8