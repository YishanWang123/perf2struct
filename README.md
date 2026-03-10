# Perf2Struct

**Performance-Conditioned Structural Generation for Fork-Shaped Beam Design**

Perf2Struct is a generative inverse design project for fork-shaped beam structures.  
The goal is to generate structural geometries from target mechanical performance indicators, and then validate the generated results through simulation-oriented engineering post-processing.

## Project Overview

![Data pipeline](assets/asset.jpg)

*Figure 1. Overall data pipeline.*

![Model](assets/assets1.jpg)

*Figure 2. Model structure.*

## Overview

This project explores how to map target structural properties to 2D fork-shaped beam geometries using conditional generative modeling.

The current workflow includes:

1. performance-conditioned structure generation
2. image-based geometry output
3. contour extraction and geometric fitting
4. COMSOL-based reconstruction and validation

## Target Conditions

The model is designed to condition on mechanical performance indicators such as:
- drive_frequency
- split
- parasitic
- x stiffness
- nonlinearity


## Project Goal

The main objective is not only to generate visually plausible structures, but also to produce candidate geometries that can serve as effective starting points for downstream simulation and optimization.

In other words, the generated image is treated as an initial design proposal, and the final performance is verified after geometric reconstruction and simulation in COMSOL.

## Method

The current research direction includes:

- conditional flow matching for structural image generation
- structured condition encoding instead of generic text encoding
- geometry reconstruction from pixel outputs
- simulation validation in COMSOL

## Install

comming soon

## train and eval

comming soon


