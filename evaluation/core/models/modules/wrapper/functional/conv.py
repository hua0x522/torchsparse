def conv3d(in_channels: int,
           out_channels: int,
           kernel_size: int = 3,
           stride: int = 1,
           dilation: int = 1,
           padding: int = 0,
           bias: bool = False,
           transpose: bool = False,
           indice_key: str = None,
           backend: str = 'torchsparse',
           kmap_mode: str = 'hashmap') -> None:
    if backend == 'torchsparse' or backend == 'torchsparse-1.4.0':
        import torchsparse.nn as spnn
        if backend == 'torchsparse':
            # return spnn.Conv3d(in_channels, out_channels, kernel_size, stride, dilation, bias, transpose, config=dict(kmap_mode=kmap_mode))
            return spnn.Conv3d(in_channels, out_channels, kernel_size, stride, dilation, bias, transpose)
        else:
            return spnn.Conv3d(in_channels, out_channels, kernel_size, stride, dilation, bias, transpose)
    elif backend == 'ME':
        import MinkowskiEngine as ME
        from .minkowski import MyMinkowskiConvolution
        if transpose:
            return ME.MinkowskiConvolutionTranspose(in_channels, out_channels, kernel_size, stride, 
                                                dilation, bias, dimension=3)
        else:
            return MyMinkowskiConvolution(in_channels, out_channels, kernel_size, stride, dilation,
                                        bias, dimension=3)
    elif backend == 'spconv':
        import spconv
        if transpose:
            return spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key,
                                            bias=bias)
        else:
            if stride == 1:
                return spconv.SubMConv3d(in_channels, out_channels, kernel_size, stride, dilation=dilation,
                                    bias=bias, indice_key=indice_key, use_hash='hash' in kmap_mode)
            else:
                return spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride, dilation=dilation, padding=padding,
                                    bias=bias, indice_key=indice_key, use_hash='hash' in kmap_mode)
    else:
        raise Exception(f"{backend} backend not supported")
