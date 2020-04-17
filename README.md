# ATReg 
Coming soon! Code review processing.
## Instruction
This code is for registering infant brain MR images. The registration is based on the segmentation map (gray matter, white matter). You can try to modify the directory in train.py to train your model. Or you can directly use test.py to test your data, where one trained model is given in ./models.

For obtaining a ACTA-Reg-Net, you need to work on ./src/demo.sh to run all the experiment.

## Requriement
- Tensorflow
- Keras

## Train
```bash
python train.py
```

## Test
```bash
python test.py
```

## Demo
```bash
cd ./src
./demo.sh -m moving.nii.gz -l moving_label.nii.gz -n save_dir
```
## Result
<img src='./Fig/Result_with_Grid.png' />
<img src='./Fig/Smoothness_Comparison.png'>

# Contact:
For any problems, please open an [issue](https://github.com/Barnonewdm/ACTA-Reg-Net/issues/new).
