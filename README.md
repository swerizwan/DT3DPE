# Text-Driven 3D Human Motion Generation for Pose Estimation using Dual-Transformer Architecture

<img src="https://camo.githubusercontent.com/2722992d519a722218f896d5f5231d49f337aaff4514e78bd59ac935334e916a/68747470733a2f2f692e696d6775722e636f6d2f77617856496d762e706e67" alt="Oryx Video-ChatGPT" data-canonical-src="https://i.imgur.com/waxVImv.png" style="max-width: 100%;">

## Overview

Text-to-motion generation has improved, but current methods struggle with realistic 3D human motions that capture both body and emotional expressions. Most focus on body motion, neglecting body language and pose estimation. We propose DT3DPE (Dual-Transformer for 3D Pose Estimation), which incorporates pose estimation and body language for more realistic, text-aligned motions. Our model uses a hierarchical quantization scheme and a dual transformer architecture for motion prediction and refinement. Experiments show DT3DPE outperforms existing methods on HumanML3D and KIT-ML datasets.

# 👁️💬 Architecture

The DT3DPE framework works as follows: (a) The input text describes an action, such as a person walking forward. (b) A movement residual is generated based on the input text. (c) The masked transformer and residual transformer process this residual to produce motion tokens and refine the motion details. (d) The final output is a detailed and coherent animation that accurately reflects the described action.

<img style="max-width: 100%;" src="https://github.com/swerizwan/DT3DPE/blob/main/resources/overview.png" alt="VERHM Overview">

# Installation

```
conda create python=3.9 --name DT3DPE
conda activate DT3DPE
```
Install the requirements
```
pip install -r requirements.txt
```
Download Pre-trained Models, Evaluation Models, and Gloves 
```
bash prepare/download_models.sh
bash prepare/download_evaluator.sh
bash prepare/download_glove.sh
```
# Demo

Output from a single prompt
```
python t2m_animation_generator.py --gpu_id 1 --ouput ouput1 --text "A person performs jumping jacks."
```
Output from a text file
```
python t2m_animation_generator.py --gpu_id 1 --ouput ouput2 --text_path ./assets/textfile.txt
```
Visualization
```
blender --background --python render.py -- --cfg=./configs/render.yaml --dir=/home/abbas/motiontext/outcomes/motiontext/HumanML3D/samples_2024-11-10-18-50-15/ --mode=video --joint_type=HumanML3D

python -m fit --dir /home/abbas/motiontext/outcomes/motiontext/HumanML3D/samples_2024-11-10-18-50-15/ --save_folder /home/abbas/motiontext/outcomes/motiontext/HumanML3D/samples_2024-11-10-18-50-15/tamp --cuda True

blender --background --python render.py -- --cfg=./configs/render.yaml --dir=/home/abbas/motiontext/results/motiontext/1222_PELearn_Diff_Latent1_MEncDec49_MdiffEnc49_bs64_clip_uncond75_01/samples_2024-10-18-22-15-14/ --mode=video --joint_type=HumanML3D
```

Qualitative results demonstrating DT3DPE's capability to synthesize human movement for pose estimation from textual descriptions.

<table>
  <tr>
    <td style="text-align: center;">
      <p>The person raises both arms up, claps their hands together,
       and takes two steps forward.</p>
      <img width="165" src="https://github.com/swerizwan/DT3DPE/blob/main/resources/1.gif" alt="Happy">
    </td>
    <td style="text-align: center;">
      <p>A person bends down, touches his toes, and stands back up.</p>
      <img width="165" src="https://github.com/swerizwan/DT3DPE/blob/main/resources/2.gif" alt="Sad">
    </td>
    <td style="text-align: center;">
      <p>The man walks forward, turns to his left, and raises both arms up.</p>
      <img width="165" src="https://github.com/swerizwan/DT3DPE/blob/main/resources/3.gif" alt="Angry">
    </td>
      <td style="text-align: center;">
      <p>A person stretches both arms out to the sides and spins around.</p>
      <img width="165" src="https://github.com/swerizwan/DT3DPE/blob/main/resources/4.gif" alt="Angry">
    </td>
  </tr>
    <tr>
    <td style="text-align: center;">
      <p>The person hops on his right foot, then lands on both feet.</p>
      <img width="165" src="https://github.com/swerizwan/DT3DPE/blob/main/resources/5.gif" alt="Happy">
    </td>
    <td style="text-align: center;">
      <p>A person kicks with right leg, kneels down, and stands back up.</p>
      <img width="165" src="https://github.com/swerizwan/DT3DPE/blob/main/resources/6.gif" alt="Frustrated">
    </td>
    <td style="text-align: center;">
      <p>The person lifts his left leg, kicks forward, and then steps back.</p>
      <img width="165" src="https://github.com/swerizwan/DT3DPE/blob/main/resources/7.gif" alt="Sad">
    </td>
    <td style="text-align: center;">
      <p>A person raises his right hand, waves, and walks away.</p>
      <img width="165" src="https://github.com/swerizwan/DT3DPE/blob/main/resources/8.gif" alt="Angry">
    </td>
  </tr>
  <tr>
    <td style="text-align: center;">
      <p>The man crouches down, reaches forward, and stands back up.</p>
      <img width="165" src="https://github.com/swerizwan/DT3DPE/blob/main/resources/9.gif" alt="Happy">
    </td>
    <td style="text-align: center;">
      <p>A person performs jumping jacks.</p>
      <img width="165" src="https://github.com/swerizwan/DT3DPE/blob/main/resources/10.gif" alt="Frustrated">
    </td>
    <td style="text-align: center;">
      <p>The person spins left in place and then raises both arms.</p>
      <img width="165" src="https://github.com/swerizwan/DT3DPE/blob/main/resources/11.gif" alt="Sad">
    </td>
    <td style="text-align: center;">
      <p>The person steps forward and walks back.</p>
      <img width="165" src="https://github.com/swerizwan/DT3DPE/blob/main/resources/12.gif" alt="Angry">
    </td>
  </tr>
</table>

## Datasets

We evaluated DT3DPE using three key datasets for text-driven human movement synthesis:

- **HumanML3D**: Combines HumanAct12 and AMASS datasets, featuring 14,616 movements and 44,970 text descriptions. It spans diverse actions like daily tasks, athletics, and performances, with clips totaling 28.59 hours. Each movement has 3-4 descriptive sentences. [Dataset Link](https://drive.google.com/file/d/1rmnG-R8wTb1sRs0PYp4RRmLg8XH-qSGW/view) 
- **KIT-ML**: Includes 3,911 movements with 6,278 text descriptions, linking human actions to natural language. It advances research on movement-language correlations with a focus on accessibility and clarity. [Dataset Link](https://drive.google.com/file/d/1IXRBm4qSjLQxp1J3cqv1xd8yb-RQY0Jz/view) 

# Train & Evaluate

- **Train**
```
python vq_trainer.py --name rvq_name --gpu_id 1 --dataset_name t2m --batch_size 256 --num_quantizers 6  --max_epoch 50 --quantize_dropout_prob 0.2 --gamma 0.05
python train_t2m_mask.py --name mtrans_name --gpu_id 2 --dataset_name t2m --batch_size 64 --vq_name rvq_name
python train_t2m_res.py --name rtrans_name  --gpu_id 2 --dataset_name t2m --batch_size 64 --vq_name rvq_name --cond_drop_prob 0.2 --share_weight
```
- **Evaluation**
```
python vq_evaluator.py --gpu_id 1 --name rvq_nq6_dc512_nc512_noshare_qdp0.2 --dataset_name t2m --ext rvq_nq6
python vq_evaluator.py --gpu_id 1 --name rvq_nq6_dc512_nc512_noshare_qdp0.2_k --dataset_name kit --ext rvq_nq6
```

