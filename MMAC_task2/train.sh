

python train.py --lesion LC --model unet++
python train.py --lesion CNV --model unet++ 
python train.py --lesion FS --model unet++ 

python train.py --lesion LC --model unet 
python train.py --lesion CNV --model unet
python train.py --lesion FS --model unet 

python train.py --lesion LC --model deeplabv3+ 
python train.py --lesion CNV --model deeplabv3+
python train.py --lesion FS --model deeplabv3+ 
