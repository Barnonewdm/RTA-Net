# An Auto-Context based Tissue-Aware Deformable Registration Network for Infant Brain MRI
Coming soon! Code review processing. Welcome for any bugs report.


## Instruction
This code is for registering infant brain MR images. The registration is based on the segmentation map (gray matter, white matter). You can try to modify the directory in train.py to train your model. Or you can directly use test.py to test your data, where one trained model is given in ./models.

For obtaining a ACTA-Reg-Net, you need to work on ./src/demo.sh to run all the experiment.

## Requriement
- Tensorflow
- Keras
- Bash

## Train
```bash
python train.py
```

## Test
```bash
python test.py gpu_id ../models/ iteration_num fixed.nii.gz moving.nii.gz moving_label.nii.gz
```

## Demo
```bash
cd ./src
./demo.sh -m moving.nii.gz -l moving_label.nii.gz -n save_dir
```
The results are saved into ../data/results/*, including warped moving image, moving label, deformation field, and displacement uncertainty map.
## Result
<img src='./Fig/Result_with_Grid.png' />
<img src='./Fig/Smoothness_Comparison.png'>

# Contact:
For any problems, please open an [issue](https://github.com/Barnonewdm/ACTA-Reg-Net/issues/new).
