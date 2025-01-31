# Efficient Connectivity-Preserving Instance Segmentation with Supervoxel-Based Loss Function

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-black)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)
![Interrogate](https://img.shields.io/badge/interrogate-84.9%25-yellow)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen?logo=codecov)
![Python](https://img.shields.io/badge/python->=3.7-blue?logo=python)

[paper](https://arxiv.org/abs/2501.01022) | poster

## Overview

This repository implements a connectivity-preserving loss function designed to improve instance segmentation of curvilinear structures. The paradigm shift here is to evaluate segmentation quality at the “structure-level” as opposed to the voxel-level. The loss is computed by detecting supervoxels in the false positive and false negative masks during training, then assigning higher penalties to *critical* supervoxels that introduce connectivity errors.

The loss computation consists of three main steps:

<blockquote>
  <p>a. <strong>Binarize Prediction</strong>: to do...</p>
  <p>b. <strong>Compute False Postive/Negative Masks</strong>: to do...</p>
  <p>c. <strong>Detect Critical Supervoxels</strong>: to do..</p>
</blockquote>
<br>

<p>
  <img src="imgs/pipeline.png" width="900" alt="pipeline">
  <br>
  <b> Figure: </b>Visualization of supervoxel-based loss computation.
</p>

## Usage

To do...

## Installation
To use the software, in the root directory, run
```bash
pip install -e .
```

## License
supervoxel-loss is licensed under the MIT License.
