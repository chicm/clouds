python train.py --encoder dpn98 --batch_size 32 --lr 0.001
python train.py --encoder densenet201 --batch_size 32 --lr 0.001
python train.py --encoder inceptionresnetv2 --batch_size 32 --lr 0.001
python train.py --encoder efficientnet-b2  --batch_size 32 --lr 0.001
python train.py --encoder senet154  --batch_size 32 --lr 0.001
python train.py --encoder dpn98 --batch_size 32 --lr 0.0001 --lrs cosine --num_epochs 40
python train.py --encoder densenet201 --batch_size 32 --lr 0.0001 --lrs cosine --num_epochs 40
python train.py --encoder inceptionresnetv2 --batch_size 32 --lr 0.0001 --lrs cosine --num_epochs 40
python train.py --encoder efficientnet-b2 --batch_size 32 --lr 0.0001 --lrs cosine --num_epochs 40
python train.py --encoder senet154 --batch_size 32 --lr 0.0001 --lrs cosine --num_epochs 40