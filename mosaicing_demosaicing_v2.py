import torch
import numpy as np
from colour_demosaicing.bayer import masks_CFA_Bayer
import torch.nn as nn
import torch.nn.functional as F


def demosaicing_CFA_Bayer_bilinear(CFA, pattern='RGGB'):
    """
    Returns the demosaiced *RGB* colourspace array from given *Bayer* CFA using
    bilinear interpolation.

    Parameters
    ----------
    CFA : array_like
        *Bayer* CFA.
    pattern : unicode, optional
        **{'RGGB', 'BGGR', 'GRBG', 'GBRG'}**,
        Arrangement of the colour filters on the pixel array.

    Returns
    -------
    ndarray
        *RGB* colourspace array.

    Notes
    -----
    -   The definition output is not clipped in range [0, 1] : this allows for
        direct HDRI / radiance image generation on *Bayer* CFA data and post
        demosaicing of the high dynamic range data as showcased in this
        `Jupyter Notebook <https://github.com/colour-science/colour-hdri/\
blob/develop/colour_hdri/examples/\
examples_merge_from_raw_files_with_post_demosaicing.ipynb>`__.

    References
    ----------
    :cite:`Losson2010c`

    Examples
    --------
    >>> import numpy as np
    >>> CFA = np.array(
    ...     [[0.30980393, 0.36078432, 0.30588236, 0.3764706],
    ...      [0.35686275, 0.39607844, 0.36078432, 0.40000001]])
    >>> demosaicing_CFA_Bayer_bilinear(CFA)
    array([[[ 0.69705884,  0.17941177,  0.09901961],
            [ 0.46176472,  0.4509804 ,  0.19803922],
            [ 0.45882354,  0.27450981,  0.19901961],
            [ 0.22941177,  0.5647059 ,  0.30000001]],
    <BLANKLINE>
           [[ 0.23235295,  0.53529412,  0.29705883],
            [ 0.15392157,  0.26960785,  0.59411766],
            [ 0.15294118,  0.4509804 ,  0.59705884],
            [ 0.07647059,  0.18431373,  0.90000002]]])
    >>> CFA = np.array(
    ...     [[0.3764706, 0.360784320, 0.40784314, 0.3764706],
    ...      [0.35686275, 0.30980393, 0.36078432, 0.29803923]])
    >>> demosaicing_CFA_Bayer_bilinear(CFA, 'BGGR')
    array([[[ 0.07745098,  0.17941177,  0.84705885],
            [ 0.15490197,  0.4509804 ,  0.5882353 ],
            [ 0.15196079,  0.27450981,  0.61176471],
            [ 0.22352942,  0.5647059 ,  0.30588235]],
    <BLANKLINE>
           [[ 0.23235295,  0.53529412,  0.28235295],
            [ 0.4647059 ,  0.26960785,  0.19607843],
            [ 0.45588237,  0.4509804 ,  0.20392157],
            [ 0.67058827,  0.18431373,  0.10196078]]])
    """

    ## Above is the original version on mosaicing_demosaicing package processing image based on numpy arrays, we adapt it to a torch tensor version as follows:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch, h, w= CFA.size()

    R_m, G_m, B_m = masks_CFA_Bayer([h, w], pattern)

    R_m = R_m[np.newaxis, np.newaxis, :]
    R_m = np.repeat(R_m, batch, axis = 0)
    G_m = G_m[np.newaxis, np.newaxis, :]
    G_m = np.repeat(G_m, batch, axis=0)
    B_m = B_m[np.newaxis, np.newaxis, :]
    B_m = np.repeat(B_m, batch, axis=0)

    R_m = torch.from_numpy(R_m).to(device)
    G_m = torch.from_numpy(G_m).to(device)
    B_m = torch.from_numpy(B_m).to(device)

    H_G = np.array(
        [[0, 1, 0],
         [1, 4, 1],
         [0, 1, 0]]) / 4  # yapf: disable

    H_G = H_G[np.newaxis, np.newaxis, :]
    H_G = torch.from_numpy(H_G).to(device)

    H_RB = np.array(
        [[1, 2, 1],
         [2, 4, 2],
         [1, 2, 1]]) / 4  # yapf: disable

    H_RB = H_RB[np.newaxis, np.newaxis, :]
    H_RB = torch.from_numpy(H_RB).to(device)
    CFA = CFA.unsqueeze(1)

    R = F.conv2d(CFA * R_m, H_RB, stride=1, padding=1)
    G = F.conv2d(CFA * G_m, H_G, stride=1, padding=1)
    B = F.conv2d(CFA * B_m, H_RB, stride=1, padding=1)

    R = R.squeeze(1)
    G = G.squeeze(1)
    B = B.squeeze(1)

    del R_m, G_m, B_m, H_RB, H_G
    torch.cuda.empty_cache()

    return torch.stack((R, G, B), dim = 3)

def mosaicing_CFA_Bayer(RGB, pattern = 'RGGB'):
    """
    Returns the *Bayer* CFA mosaic for a given *RGB* colourspace array.

    Parameters
    ----------
    RGB : array_like
        *RGB* colourspace array.
    pattern : unicode, optional
        **{'RGGB', 'BGGR', 'GRBG', 'GBRG'}**,
        Arrangement of the colour filters on the pixel array.

    Returns
    -------
    ndarray
        *Bayer* CFA mosaic.

    Examples
    --------
    >>> import numpy as np
    >>> RGB = np.array([[[0, 1, 2],
    ...                  [0, 1, 2]],
    ...                 [[0, 1, 2],
    ...                  [0, 1, 2]]])
    >>> mosaicing_CFA_Bayer(RGB)
    array([[ 0.,  1.],
           [ 1.,  2.]])
    >>> mosaicing_CFA_Bayer(RGB, pattern='BGGR')
    array([[ 2.,  1.],
           [ 1.,  0.]])
    """

    ## Above is the original version on mosaicing_demosaicing package processing image based on numpy arrays, we adapt it to a torch tensor version as follows:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    R = RGB[:, :, :, 0]
    G = RGB[:, :, :, 1]
    B = RGB[:, :, :, 2]

    batch, _, _, _ = RGB.shape
    R_m, G_m, B_m = masks_CFA_Bayer(RGB.shape[1:3], pattern)

    G_m = G_m[np.newaxis, :]
    G_m = np.repeat(G_m, batch, axis = 0)
    B_m = B_m[np.newaxis, :]
    B_m = np.repeat(B_m, batch, axis = 0)
    R_m = R_m[np.newaxis, :]
    R_m = np.repeat(R_m, batch, axis = 0)

    R_m = torch.from_numpy(R_m).to(device)
    G_m = torch.from_numpy(G_m).to(device)
    B_m = torch.from_numpy(B_m).to(device)

    CFA = R * R_m + G * G_m + B * B_m
    del R_m, G_m, B_m
    torch.cuda.empty_cache()

    return CFA

