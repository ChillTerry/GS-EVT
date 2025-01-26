# GS-EVT
## How to install
```sh
git clone https://github.com/ChillTerry/GS-EVT.git --recursive
cd GSEVT
conda create --name GSEVT python=3.7
conda activate GSEVT
pip install -r requirements.txt
cd submodules/diff-gaussian-rasterization/
pip install -e .
```

## How to get the dataset
We have preprocessed the [VECTOR](https://star-datasets.github.io/vector/) dataset for you, so you can download it [here](https://drive.google.com/drive/folders/1NShkkljDLqleAGy-goFen3K0C102nIo3?usp=drive_link).

## How to run a sequence
```sh
python main.py -c configs/VECTOR/desk_normal1_config.yaml
```

## Structure of this project
```
GSEVT
├── configs                                 // Store config yamls for each sequence
├── data                                    // Store data of gs-splat maps and event data(in txt) 
├── gaussian_splatting                      // original gaussian splatting modules
├── gui                                     // gui 
├── results                                 // tracking outputs
├── scripts                                 // scripts for debug on server
│   ├── desk_batch_run.sh
│   ├── robot_batch_run.sh
│   └── sofa_batch_run.sh
├── submodules
│   └── diff-gaussian-rasterization         // customized rasterization module for this project
├── test                                    // test scripts
│   ├── test_event_camera.py
│   ├── test_event_rgb_overlay.py
│   ├── test_initial_pose.py
│   ├── test_intensity_change.py
│   └── test_single_frame_tracking.py
├── utils
│   ├── auxiliary.py
│   ├── event_camera
│   ├── pose.py                             // functions for calculating pose
│   ├── render_camera                       // defines Camera class and RenderFrame class
│   ├── tracker.py                          // code for tracking
│   └── visualizer.py                       // code for visualing tracking results
├── main.py                                 // Run this file to start the tracking process
├── LICENSE
├── readme.txt
└── requirements.txt                        // packages needed for this project
```
