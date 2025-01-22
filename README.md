# Efficient Connectivity-Preserving Instance Segmentation with Supervoxel-Based Loss Function

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-black)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)
![Interrogate](https://img.shields.io/badge/interrogate-61.8%25-red)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen?logo=codecov)
![Python](https://img.shields.io/badge/python->=3.7-blue?logo=python)

[paper](https://arxiv.org/abs/2501.01022) | poster

## Overview

This repository implements a connectivity-preserving loss function designed to address the challenge of performing accurate instance segmentation of neuron morphologies. The method uses a topological loss function that leverages the concept of "supervoxels"—connected sets of voxels—to detect and penalize connectivity errors during training. This approach efficiently handles the segmentation of curvilinear, filamentous structures, with minimal computational overhead.


## Compute Loss

To do...

## Installation
To use the software, in the root directory, run
```bash
pip install -e .
```
