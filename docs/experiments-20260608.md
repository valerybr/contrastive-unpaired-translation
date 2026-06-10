# Experiments summary - 2026-06-08
### Summary
1. Stable training
2. No difference between bidirectrional training and fastcut for L->R. Both - with masks<br>
2.1. Need to test R->L

## Disable idt loss

1. Use FastCUT model with default weight $\lambda$(PatchNCE) = 10
2. Use FastCUT model with default weight $\lambda$(PatchNCE) = 5

## Use masks and modify weights

Create masks during the preprocess
* Modify GAN loss to use masked areas only
* Modify PatchNCE loss to sample patches from masked loss only

1. Use masks with hard step edge -> created artefacts
2. Use edge bluring -> fixed the artefacts problem


## Bidirectional training 
1. Modify Loss functions - need to analyze images


## Results viewer 
1. Add findings
2. Zoom 


# Next steps
1. Add lesion detection - positive / negative
2. Should I use full resolution and tile images - calcifications can be lossed otherwise
3. metrics for model
4. MLO view 

