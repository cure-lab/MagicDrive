## Note for xformers installation

As bevfusion (or mmdet3d) works with `Pytorch==1.10.2`, we opt to keep the version of pytorch. However, xformers does not officially support `Pytorch<2.0.0`. Therefore, we made a customized version of xformers with `Pytorch==1.10.2`. We only change some searching path of Pytorch, it should be safe.

> Only tested with cuda10.2 on V100. `flash attention` is not supported on V100, therefore our xformers will ignore flash attention on installation.

> [!NOTE]  
> Please DO NOT use `triton` or `flash-attn` with this `xformers`. If you have installed `triton` in your python environments, please uninstall it.
