
# Code for Zero-Shot Instance Segmentation with RoI-wise Background Transformers


## Code requirements
+ python: python3.7
+ nvidia GPU
+ pytorch1.1.0
+ GCC >=5.4
+ NCCL 2
+ the other python libs in requirement.txt

## Install 

```
conda create -n RB-ZSI python=3.7 -y
conda activate RB-ZSI

conda install pytorch=1.1.0 torchvision=0.3.0 cudatoolkit=10.0 -c pytorch

pip install cython && pip --no-cache-dir install -r requirements.txt
   
python setup.py develop
```

## Dataset prepare


- Download the train and test annotations files for RB-ZSI from [annotations](https://drive.google.com/drive/folders/1TLbmDoRiKcMGq1zyVahXtGVTdkvI9Dus?usp=sharing), put all json label file to

    The annotation file is provided by [Zero-shot-Instance-Segmentation](https://github.com/zhengye1995/Zero-shot-Instance-Segmentation) [1]
    ```
    data/coco/annotations/
    ```
    


- Download MSCOCO-2014 dataset and unzip the images it to path： 
    ```
    data/coco/train2014/
    data/coco/val2014/
    ```


- **Training**:
(the checkpoint files will be saved at work_dir folder by default )
     - 48/17 split:
       ```
       chmod +x tools/dist_train.sh
       ./tools/dist_train.sh configs/RB-ZSI/48_17/train/RB-ZSI.py 2
       ```
        
    - 65/15 split:
      ```
      chmod +x tools/dist_train.sh
      ./tools/dist_train.sh configs/RB-ZSI/65_15/train/RB-ZSI.py 2
      ```
      
          
- **Inference & Evaluate**:
    (need to create a checkpoints folder at first)

    + **RB-ZSI task**:

        - 48/17 split RB-ZSI task:
           
            
            - inference:
                ```
                chmod +x tools/dist_test.sh
                ./tools/dist_test.sh configs/RB-ZSI/48_17/test/zsi/RB-ZSI.py checkpoints/RB-ZSI-selector-2_48_17.pth 4 --json_out results/RB-ZSI-selector-2_48_17.json
                ```
                
            - evaluate:
                - for RB-ZSD performance
                    ```
                    python tools/rb-zsi_coco_eval.py results/RB-ZSI-selector-2_48_17.bbox.json --ann data/coco/annotations/instances_val2014_unseen_48_17.json
                    ```
                - for RB-ZSI performance
                    ```
                    python tools/rb-zsi_coco_eval.py results/RB-ZSI-selector-2_48_17.segm.json --ann data/coco/annotations/instances_val2014_unseen_48_17.json --types segm
                    ```
        - 65/15 split RB-ZSI task:
           
            - inference:
                ```
                chmod +x tools/dist_test.sh
                ./tools/dist_test.sh configs/RB-ZSI/65_15/test/zsi/RB-ZSI.py checkpoints/RB-ZSI-selector-2_65_15.pth 4 --json_out results/RB-ZSI-selector-2_65_15.json
                ```
            
            - evaluate:
                - for RB-ZSD performance
                    ```
                    python tools/rb-zsi_coco_eval.py results/RB-ZSI-selector-2_65_15.bbox.json --ann data/coco/annotations/instances_val2014_unseen_65_15.json
                    ```
                - for RB-ZSI performance
                    ```
                    python tools/rb-zsi_coco_eval.py results/RB-ZSI-selector-2_65_15.segm.json --ann data/coco/annotations/instances_val2014_unseen_65_15.json --types segm
                    ```

    + **GZSI task**:

        - 48/17 split RB-GZSI task:
            - use the same model file RB-ZSI-selector-2_48_17.pth in RB-ZSI task   
            - inference:
                ```
                chmod +x tools/dist_test.sh
                ./tools/dist_test.sh configs/RB-ZSI/48_17/test/gzsi/RB-GZSI.py checkpoints/RB-ZSI-selector-2_48_17.pth 4 --json_out results/RB-GZSI-selector-2_48_17.json
                ```
            
            - evaluate:
                - for gzsd
                    ```
                    python tools/rb-gzsi_coco_eval.py results/RB-GZSI-selector-2_48_17.bbox.json --ann data/coco/annotations/instances_val2014_gzsi_48_17.json --gzsi --num-seen-classes 48
                    ```
                - for gzsi
                    ```
                    python tools/gzsi_coco_eval.py results/RB-GZSI-selector-2_48_17.segm.json --ann data/coco/annotations/instances_val2014_gzsi_48_17.json --gzsi --num-seen-classes 48 --types segm
                    ```
        - 65/15 split RB-GZSI task:
            - use the same model file RB-ZSI-selector-2_65_15.pth in RB-ZSI task   
            - inference:
                ```
                chmod +x tools/dist_test.sh
                ./tools/dist_test.sh configs/RB-ZSI/65_15/test/gzsi/RB-GZSI.py checkpoints/RB-ZSI-selector-2_65_15.pth 4 --json_out results/RB-GZSI-selector-2_65_15.json
                ```
            
            - evaluate:
                - for gzsd
                    ```
                    python tools/rb-gzsi_coco_eval.py results/RB-GZSI-selector-2_65_15.bbox.json --ann data/coco/annotations/instances_val2014_gzsi_65_15.json --gzsi --num-seen-classes 65
                    ```
                - for gzsi
                    ```
                    python tools/rb-gzsi_coco_eval.py results/RB-GZSI-selector-2_65_15.segm.json --ann data/coco/annotations/instances_val2014_gzsi_65_15.json --gzsi --num-seen-classes 65 --types segm
                    ```
# Credits
Our code makes modification from the following sources:
- [1] Zero-shot-Instance-Segmentation github repository: https://github.com/zhengye1995/Zero-shot-Instance-Segmentation
- [2] Zero-shot-Instance-Segmentation paper: https://openaccess.thecvf.com/content/CVPR2021/papers/Zheng_Zero-Shot_Instance_Segmentation_CVPR_2021_paper.pdf



