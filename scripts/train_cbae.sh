OUT_DIR=""

# Check if the first argument is --out_dir
if [[ "$1" == "--out_dir" && -n "$2" ]]; then
    OUT_DIR="$2"
    shift 2
fi

# Build the argument string
OUT_ARG=""
if [ -n "$OUT_DIR" ]; then
    OUT_ARG="--out-dir \"$OUT_DIR\""
fi

### training CB-AE for CelebA-HQ-pretrained StyleGAN2 with supervised models as pseudolabeler
python -u train/train_cbae_gan.py -d celebahq40 -e cbae_stygan2 -p supervised -s 40concepts $OUT_ARG

### training CC for CelebA-HQ-pretrained StyleGAN2 with supervised models as pseudolabeler
# python3 -u train/train_cc_gan.py -e cc_stygan2_thr90 -p supervised -t sup_pl_cls8