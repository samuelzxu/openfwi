# Iteration 0

## Idea

Currently the dataset is in shards of 500, each in a numpy file. Moreover, the dataset is stored online on google drive :/
The existing dataloader seems to draw from these stacks of 500 rather inefficiently, and moreover it trains only one st
For this iteration I want to use the existing OpenFWI base inversionNet, except with improved data sampling.

## Goals

- Run InversionNet with reasonable batch size (192)
- Train with precise data sampling

## Motivation

## Implemenation Plan

## Results

87 on Public LB
Solid.