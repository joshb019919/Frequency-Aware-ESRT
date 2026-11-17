## Command

- se the command `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python main.py --train --skip_download --hf_repo mAiello00/DIV2K --data_dir /data/data/div2k -- -nEpochs 1`.
- Do `python scripts/png2npy.py --pathFrom <images-to-convert> --pathTo <folder-for-npy-images>` for each set of images.
    - Expects as mentioned below.

## Datasets and Expected Structures

Maybe download and extract the raw PNG files to a DIV2K_raw folder?

- Get Set5 dataset from https://github.com/linzino7/Set5-dataset's HR folder.
    - Expects it in `Test_Datasets/Set5/`.
- Get Set5_LR dataset from the same above but LR folder.
    - Expects it in `Test_Datasets/Set5_LR/x<scale>`.
- Get HR training images from http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip.
    - Expects it in `DF2K_decoded/DIV2K_train_HR/`.
- Get LR downscaled training images from http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X<scale>.zip
    - Expects it in `DF2K_decoded/DIV2K_train_LR_bicubic/X<scale>`.
    - Need at least one scale, e.g., 2.

## Fixing Errors

- Do `unset LD_LIBRARY_PATH` in case of error: 
    - `RuntimeError: CUDNN_BACKEND_TENSOR_DESCRIPTOR cudnnFinalize failedptrDesc->finalize() cudnn_status: CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED`.
    - For some reason, it was bleeding versions from another virtual environment on my machine.

## Notes

- Scales must match between training and Set5 images.
- Transforming Set5 images to .npy is probably unnecessary since it's only 5 of them.
- Be careful with patch size.  Default is 192, but this is intended for $\times 4$ downscaled; $\times 2$ downscaled suggests 96.
    - 192 patch size gives $96\times 96$ patches, which is big memory.
    - Can also make far fewer batches, 4 instead of default 16, or only use $\times 4$ downscaled images.
