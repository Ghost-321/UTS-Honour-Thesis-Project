# Super-Resolution for Zero-Shot Diabetic Retinopathy Classification

This repository contains the code used for the Honours thesis:
*“Investigating the Influence of Super-Resolution on Zero-Shot Diabetic Retinopathy Classification (UTS, 2025)”*

The project evaluates whether improving retinal image visibility using conservative super-resolution can enhance zero-shot diabetic retinopathy classification without additional labelled training data. For additional Artefacts visit this One-Drive:

https://studentutsedu-my.sharepoint.com/:u:/g/personal/jonathan_morel_student_uts_edu_au/ESVDDdTVQ9JMg2ddO67SQOcBDOo2QOa6KTceYChUPjaFcQ?e=ehDQbb

## Contents
- Super-resolution models (Bicubic, SRCNN, FSRCNN, ESPCN)
- Zero-shot classification (CLIP + RETCLIP)
- Retinal degradation simulation
- Configuration files for reproducibility
- Notebooks for analysis and visualisation

## Setup
Install dependencies and update dataset paths in:
`configs/config.yaml`

## Running
Super-resolution:
`python super_resolution/run_sr.py`

Zero-shot evaluation:
`python zsl_evaluation/run_zsl.py`

Outputs are saved in:
`./outputs/`

## Notes
- Datasets are not included in this repository
- Academic use only



