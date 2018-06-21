#File structure

- `dev_main256.py` is the file used to read in numpy arrays on summitdev for image size 256. It reads data using `data.hvd_loaders` file and uses `hj_fp16.py` model which HJ wrote for half precision. Models can be swapped easily. 
- `hvd_main256.py`is the file used to work on 256 image size by using the same `data.hvd_loaders` file but the only thing that needs to be changed in `data.hvd_loader` is replace `from data.data_npy import ChestXrayDataSet` to `from data.data import ChestXrayDataSets`. This is going to read in images instead of images on Summit

**Note: Any file with a prefix `hvd` uses horovod**




# Data parallel

Note: `any file with hvd corresponds to horovod implementation`

## RESULTS for Summit

### Horovod with n1a6g6 configuration

- Data size = 4096
- Batch size = 32
- number of epochs = 3

| Nodes | First epoch | Subsequent epochs | Total time |
|-------|-------------|-------------------|------------|
|  1    |   170       |  153      | 478 |
|  4    |    71       |   51      | 176 |
|  16   |    60       |   17      |  95 |



### Horovod with n1a1g6 configuration with nn.DataParallel()

- Data size = 4096
- Batch size = 32
- number of epochs = 3

| Nodes | First epoch | Subsequent epochs | total time |
|-------|-------------|-------------------|------------|
|  1    |     130     |       94    |   319  |
|  2    |    


## RESULTS for Summit-dev

### Horovod with n1a6g6 configuration

- Data size = 4096
- Batch size = 32
- number of epochs = 3

| Nodes | First epoch | Subsequent epochs | Total time |
|-------|-------------|-------------------|------------|
|  1    |  
|  4    |   
|  16   |  
|  32   |



### Horovod with n1a1g6 configuration with nn.DataParallel()

- Data size = 4096
- Batch size = 32
- number of epochs = 3

| Nodes | First epoch | Subsequent epochs | total time |
|-------|-------------|-------------------|------------|
|  1    |  
|  4    |   
|  16   |  
|  32   |
