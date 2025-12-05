# SMPL Reconstruction from 3D Skeleton Joints

This repository provides a flexible pipeline for converting **3D skeleton joints** into their corresponding **SMPLX mesh representations**. It supports skeleton formats that are not based on OpenPose, such as the 51-joint Aria Digital Twin or the 25-joint NTU RGB+D format.

To run the reconstruction, you only need two components:
1. Place a sequence of 3D skeleton joints with the shape **(num_frames, num_joints, 3)** in the `sample` folder.
2. Provide a **joint mapping** between your skeleton format and the SMPLX joint set. You can modify this mapping in the `data_parser.py` file.

With these inputs, the code recovers an SMPLX human mesh for each frame in the sequence.

This implementation is based on [smplify-x](https://github.com/vchoutas/smplify-x). If you need further details, please refer to the original repository. Many thanks to [smplify-x](https://github.com/vchoutas/smplify-x) for their foundational work and contributions.


## 📋 Contents
- [Prerequistes](#-prerequistes)
- [Project Structure](#-project_structure)
- [Fitting SMPLX](#-fitting)
- [Visualization](#-visualization)

## 1. Prerequistes
<a id="-prerequistes"></a>
Before you convert the 3D skeleton joints into the SMPLX mesh, please install dependencies and download necessary models required by SMPLify-X following the below procedures. In addition, you can also refer to the [smplify-x](https://github.com/vchoutas/smplify-x) repository for more details.


### 1.1 Create a new environment
```bash
conda create -n smpl python=3.12 # create a new environment for SMPL
```

### 1.2 Install [SMPLX](https://github.com/vchoutas/smplx) dependency
```bash
mkdir && cd dependencies
pip install smplx[all]
git clone https://github.com/vchoutas/smplx
cd smplx
python setup.py install
cd ..
```

### 1.3 Install [VPoser](https://github.com/nghorbani/HumanBodyPrior) dependency
```bash
git clone https://github.com/nghorbani/human_body_prior
```
Please comment out the `line 39-41` of the `human_body_prior/setup.py` first and then run the below command.
```bash
cd human_body_prior
git checkout cvpr19
python setup.py develop
cd ..
```

### 1.4 Install [torch-mesh-isect](https://github.com/vchoutas/torch-mesh-isect) dependency
```bash
git clone https://github.com/vchoutas/torch-mesh-isect
cd torch-mesh-isect
python setup.py install
cd ..
```
> Download the `models_smplx_v1_1.zip` and `vposer_v1_0.zip` files from the [SMPL website](https://smpl-x.is.tue.mpg.de/) and ***unzip*** them into this folder.

### ⚙️ Additional Fixes for Common Build Errors
<details>
<summary>Click to expand fixes and errors</summary>

1. If the error `src/bvh.cpp:26:23: error: ‘AT_CHECK’ was not declared in this scope; did you mean ‘CHECK’?` appears, please add `#define AT_CHECK TORCH_CHECK` to the `torch-mesh-isect/src/bvh.cpp` file.
2. If the error `src/bvh_cuda_op.cu:38:10: fatal error: helper_math.h: No such file or directory` appears, please download `helper_math.h` from the [CUDA Samples repository](https://github.com/NVIDIA/cuda-samples/tree/master/Common) and place it in the `torch-mesh-isect/src` directory.
3. If the error `src/bvh_cuda_op.cu(945): error: no suitable conversion function from "const at::DeprecatedTypeProperties" to "c10::ScalarType" exists` appears, please modify `triangles.type()` on line 946 in `torch-mesh-isect/src/bvh_cuda_op.cu` to `triangles.scalar_type()`.
4. If the error `RuntimeError: Subtraction, the `-` operator, with a bool tensor is not supported. If you are trying to invert a mask, use the `~` or `logical_not()` operator instead.` appears, please update **torchgeometry** file via mannually change the `conversions.py` file in `torchgeometry/core`. You can find it in `/home/your_username/.conda/envs/smpl/lib/python3.12/site-packages/torchgeometry/core/conversions.py` and replace `lines 301-304` by the following:
    ```bash
    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * ~(mask_d0_d1)
    mask_c2 = ~(mask_d2) * mask_d0_nd1
    mask_c3 = ~(mask_d2) * ~(mask_d0_nd1)
    ```
</details>


## 2. Project structure
<a id="-project_structure"></a>

```bash
smplifyx-skeleton
  ├──cfg_files  
  ├──models  # from `models_smplx_v1_1.zip`
  │   └──smplx 
  │        ├──SMPLX_FEMALE.npz
  │        ├──SMPLX_FEMALE.pkl
  │        └── .....   
  ├──optimizers  
  ├──output_folder
  ├──samples  # you can replace it with your own data
  ├──vposer_v1_0   # from `vposer_v1_0.zip`
  ├──dependencies  
  │   ├── human_body_prior   
  │   ├── smplx
  │   └── torch-mesh-isect 
  └──....   # other python files
 
```

## 3. Fitting SMPLX
<a id="-fitting"></a>
```Shell
python main.py --config cfg_files/fit_smplx.yaml 
  --data_folder samples/sequence1
  --output_folder output_folder 
  --dataset custom 
```

## 4. Visualization
<a id="-visualization"></a>
To visualize the skeleton joints and the corresponding SMPLX mesh, please refer to the `visualization.ipynb` file.

## Acknowledgements
This repo is based on [smplify-x](https://github.com/vchoutas/smplify-x). Thanks to the authors for their work!







