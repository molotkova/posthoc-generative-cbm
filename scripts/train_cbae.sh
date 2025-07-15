out_dir=""
job_id=""

# Parse arguments in any order
while [[ $# -gt 0 ]]; do
    case $1 in
        --out-dir)
            if [[ -n "$2" ]]; then
                out_dir="$2"
                shift 2
            else
                echo "Error: --out-dir requires a directory argument."
                exit 1
            fi
            ;;
        --job-id)
            if [[ -n "$2" ]]; then
                job_id="$2"
                shift 2
            else
                echo "Error: --job-id requires an argument."
                exit 1
            fi
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Build the argument strings
OUT_ARG=""
if [ -n "$out_dir" ]; then
    OUT_ARG="--out-dir $out_dir"
fi

JOB_ARG=""
if [ -n "$job_id" ]; then
    JOB_ARG="--job-id $job_id"
fi

### training CB-AE for CelebA-HQ-pretrained StyleGAN2 with supervised models as pseudolabeler
python -u train/train_cbae_gan.py -d celebahq40 -e cbae_stygan2_thr0 -p clipzs -t sup_cls40 $OUT_ARG $JOB_ARG

### training CC for CelebA-HQ-pretrained StyleGAN2 with supervised models as pseudolabeler
# python3 -u train/train_cc_gan.py -e cc_stygan2_thr90 -p supervised -t sup_pl_cls8