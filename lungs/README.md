# Data parallel

Note: `any file with hvd corresponds to horovod implementation`

## RESULTS

### Horovod with n1a6g6 configuration

- Data size = 4096
- Batch size = 32
- number of epochs = 4

| Nodes | First epoch | Subsequent epochs | Total time |
|-------|-------------|-------------------|------------|
|  1    |   170       |  153      | 478 |
|  4    |    71       |   51      | 176 |
|



### Horovod with n1a1g6 configuration with nn.DataParallel()

- Data size = 4096
- Batch size = 32
- number of epochs = 4

| Nodes | First epoch | Subsequent epochs | total time |
|-------|-------------|-------------------|------------|
|  1    |     130     |       94    |   319  |
|  2    |    


